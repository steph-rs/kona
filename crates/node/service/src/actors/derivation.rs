//! [NodeActor] implementation for the derivation sub-routine.

use crate::{Metrics, NodeActor};
use async_trait::async_trait;
use kona_derive::{
    ActivationSignal, Pipeline, PipelineError, PipelineErrorKind, ResetError, ResetSignal, Signal,
    SignalReceiver, StepResult,
};
use kona_protocol::{BlockInfo, L2BlockInfo, OpAttributesWithParent};
use thiserror::Error;
use tokio::{
    select,
    sync::{mpsc, oneshot, watch},
};
use tokio_util::sync::CancellationToken;

/// The [NodeActor] for the derivation sub-routine.
///
/// This actor is responsible for receiving messages from [NodeActor]s and stepping the
/// derivation pipeline forward to produce new payload attributes. The actor then sends the payload
/// to the [NodeActor] responsible for the execution sub-routine.
#[derive(Debug)]
pub struct DerivationActor<P>
where
    P: Pipeline + SignalReceiver,
{
    /// The derivation pipeline.
    pipeline: P,

    /// The l2 safe head from the engine.
    engine_l2_safe_head: watch::Receiver<L2BlockInfo>,
    /// A receiver that tells derivation to begin. Completing EL sync consumes the instance.
    el_sync_complete_rx: oneshot::Receiver<()>,
    /// A receiver that sends a [`Signal`] to the derivation pipeline.
    ///
    /// The derivation actor steps over the derivation pipeline to generate
    /// [`OpAttributesWithParent`]. These attributes then need to be executed
    /// via the engine api, which is done by sending them through the
    /// `attributes_out` channel.
    ///
    /// When the engine api receives an `INVALID` response for a new block (
    /// the new [`OpAttributesWithParent`]) during block building, the payload
    /// is reduced to "deposits-only". When this happens, the channel and
    /// remaining buffered batches need to be flushed out of the derivation
    /// pipeline.
    ///
    /// This channel allows the engine to send a [`Signal::FlushChannel`]
    /// message back to the derivation pipeline when an `INVALID` response
    /// occurs.
    ///
    /// Specs: <https://specs.optimism.io/protocol/derivation.html#l1-sync-payload-attributes-processing>
    derivation_signal_rx: mpsc::Receiver<Signal>,
    /// The receiver for L1 head update notifications.
    l1_head_updates: watch::Receiver<Option<BlockInfo>>,

    /// The sender for derived [`OpAttributesWithParent`]s produced by the actor.
    attributes_out: mpsc::Sender<OpAttributesWithParent>,
    /// The reset request sender, used to handle [`PipelineErrorKind::Reset`] events and forward
    /// them to the engine.
    reset_request_tx: mpsc::Sender<()>,

    /// A flag indicating whether or not derivation is idle. Derivation is considered idle when it
    /// has yielded to wait for more data on the DAL.
    derivation_idle: bool,
    /// A flag indicating whether or not derivation is waiting for a signal. When waiting for a
    /// signal, derivation cannot process any incoming events.
    waiting_for_signal: bool,

    /// The cancellation token, shared between all tasks.
    cancellation: CancellationToken,
}

impl<P> DerivationActor<P>
where
    P: Pipeline + SignalReceiver,
{
    /// Creates a new instance of the [DerivationActor].
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        pipeline: P,
        engine_l2_safe_head: watch::Receiver<L2BlockInfo>,
        el_sync_complete_rx: oneshot::Receiver<()>,
        derivation_signal_rx: mpsc::Receiver<Signal>,
        l1_head_updates: watch::Receiver<Option<BlockInfo>>,
        attributes_out: mpsc::Sender<OpAttributesWithParent>,
        reset_request_tx: mpsc::Sender<()>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            pipeline,
            engine_l2_safe_head,
            el_sync_complete_rx,
            derivation_signal_rx,
            l1_head_updates,
            attributes_out,
            reset_request_tx,
            derivation_idle: true,
            waiting_for_signal: false,
            cancellation,
        }
    }

    /// Handles a [`Signal`] received over the derivation signal receiver channel.
    async fn signal(&mut self, signal: Signal) {
        if let Signal::Reset(ResetSignal { l1_origin, .. }) = signal {
            kona_macros::set!(counter, Metrics::DERIVATION_L1_ORIGIN, l1_origin.number);
        }

        match self.pipeline.signal(signal).await {
            Ok(_) => info!(target: "derivation", ?signal, "[SIGNAL] Executed Successfully"),
            Err(e) => {
                error!(target: "derivation", ?e, ?signal, "Failed to signal derivation pipeline")
            }
        }
    }

    /// Attempts to step the derivation pipeline forward as much as possible in order to produce the
    /// next safe payload.
    async fn produce_next_attributes(&mut self) -> Result<OpAttributesWithParent, DerivationError> {
        // As we start the safe head at the disputed block's parent, we step the pipeline until the
        // first attributes are produced. All batches at and before the safe head will be
        // dropped, so the first payload will always be the disputed one.
        loop {
            let l2_safe_head = *self.engine_l2_safe_head.borrow();
            match self.pipeline.step(l2_safe_head).await {
                StepResult::PreparedAttributes => { /* continue; attributes will be sent off. */ }
                StepResult::AdvancedOrigin => {
                    let origin =
                        self.pipeline.origin().ok_or(PipelineError::MissingOrigin.crit())?.number;

                    kona_macros::set!(counter, Metrics::DERIVATION_L1_ORIGIN, origin);
                    debug!(target: "derivation", l1_block = origin, "Advanced L1 origin");
                }
                StepResult::OriginAdvanceErr(e) | StepResult::StepFailed(e) => {
                    match e {
                        PipelineErrorKind::Temporary(e) => {
                            // NotEnoughData is transient, and doesn't imply we need to wait for
                            // more data. We can continue stepping until we receive an Eof.
                            if matches!(e, PipelineError::NotEnoughData) {
                                continue;
                            }

                            debug!(
                                target: "derivation",
                                "Exhausted data source for now; Yielding until the chain has extended."
                            );
                            return Err(DerivationError::Yield);
                        }
                        PipelineErrorKind::Reset(e) => {
                            warn!(target: "derivation", "Derivation pipeline is being reset: {e}");

                            let system_config = self
                                .pipeline
                                .system_config_by_number(l2_safe_head.block_info.number)
                                .await?;

                            if matches!(e, ResetError::HoloceneActivation) {
                                let l1_origin = self
                                    .pipeline
                                    .origin()
                                    .ok_or(PipelineError::MissingOrigin.crit())?;
                                self.pipeline
                                    .signal(
                                        ActivationSignal {
                                            l2_safe_head,
                                            l1_origin,
                                            system_config: Some(system_config),
                                        }
                                        .signal(),
                                    )
                                    .await?;
                            } else {
                                if let ResetError::ReorgDetected(expected, new) = e {
                                    warn!(
                                        target: "derivation",
                                        "L1 reorg detected! Expected: {expected} | New: {new}"
                                    );

                                    kona_macros::inc!(counter, Metrics::L1_REORG_COUNT);
                                }

                                self.reset_request_tx.send(()).await.map_err(|e| {
                                    error!(target: "derivation", ?e, "Failed to send reset request");
                                    DerivationError::Sender(Box::new(e))
                                })?;
                                self.waiting_for_signal = true;
                                return Err(DerivationError::Yield);
                            }
                        }
                        PipelineErrorKind::Critical(_) => {
                            error!(target: "derivation", "Critical derivation error: {e}");
                            kona_macros::inc!(counter, Metrics::DERIVATION_CRITICAL_ERROR);
                            return Err(e.into());
                        }
                    }
                }
            }

            // If there are any new attributes, send them to the execution actor.
            if let Some(attrs) = self.pipeline.next() {
                return Ok(attrs);
            }
        }
    }
}

#[async_trait]
impl<P> NodeActor for DerivationActor<P>
where
    P: Pipeline + SignalReceiver + Send + Sync,
{
    type InboundEvent = InboundDerivationMessage;
    type Error = DerivationError;

    async fn start(mut self) -> Result<(), Self::Error> {
        loop {
            select! {
                biased;

                _ = self.cancellation.cancelled() => {
                    info!(
                        target: "derivation",
                        "Received shutdown signal. Exiting derivation task."
                    );
                    return Ok(());
                }
                signal = self.derivation_signal_rx.recv() => {
                    let Some(signal) = signal else {
                        error!(
                            target: "derivation",
                            ?signal,
                            "DerivationActor failed to receive signal"
                        );
                        return Err(DerivationError::SignalReceiveFailed);
                    };

                    self.signal(signal).await;
                    self.waiting_for_signal = false;
                }
                msg = self.l1_head_updates.changed() => {
                    if let Err(err) = msg {
                        error!(
                            target: "derivation",
                            ?err,
                            "L1 head update stream closed without cancellation. Exiting derivation task."
                        );
                        return Ok(());
                    }

                    self.process(InboundDerivationMessage::NewDataAvailable).await?;
                }
                _ = self.engine_l2_safe_head.changed() => {
                    self.process(InboundDerivationMessage::SafeHeadUpdated).await?;
                }
                _ = &mut self.el_sync_complete_rx, if !self.el_sync_complete_rx.is_terminated() => {
                    info!(target: "derivation", "Engine finished syncing, starting derivation.");
                    // Optimistically process the first message.
                    self.process(InboundDerivationMessage::NewDataAvailable).await?;
                }
            }
        }
    }

    /// Attempts to process the next payload attributes.
    ///
    /// There are a few constraints around stepping on the derivation pipeline.
    /// - The l2 safe head ([`L2BlockInfo`]) must not be the zero hash.
    /// - The pipeline must not be stepped on with the same L2 safe head twice.
    /// - Errors must be bubbled up to the caller.
    ///
    /// In order to achieve this, the channel to receive the L2 safe head
    /// [`L2BlockInfo`] from the engine is *only* marked as _seen_ after payload
    /// attributes are successfully produced. If the pipeline step errors,
    /// the same [`L2BlockInfo`] is used again. If the [`L2BlockInfo`] is the
    /// zero hash, the pipeline is not stepped on.
    async fn process(&mut self, msg: Self::InboundEvent) -> Result<(), Self::Error> {
        // Only attempt derivation once the engine finishes syncing.
        if !self.el_sync_complete_rx.is_terminated() {
            trace!(target: "derivation", "Engine not ready, skipping derivation");
            return Ok(());
        } else if self.waiting_for_signal {
            trace!(target: "derivation", "Waiting to receive a signal, skipping derivation");
            return Ok(());
        }

        // If derivation isn't idle and the message hasn't observed a safe head update already,
        // check if the safe head has changed before continuing. This is to prevent attempts to
        // progress the pipeline while it is in the middle of processing a channel.
        if !(self.derivation_idle || msg == InboundDerivationMessage::SafeHeadUpdated) {
            match self.engine_l2_safe_head.has_changed() {
                Ok(true) => { /* Proceed to produce next payload attributes. */ }
                Ok(false) => {
                    trace!(target: "derivation", "Safe head hasn't changed, skipping derivation.");
                    return Ok(());
                }
                Err(e) => {
                    error!(target: "derivation", ?e, "Failed to check if safe head has changed");
                    return Err(DerivationError::L2SafeHeadReceiveFailed);
                }
            }
        }

        // Wait for the engine to initialize unknowns prior to kicking off derivation.
        let engine_safe_head = *self.engine_l2_safe_head.borrow();
        if engine_safe_head.block_info.hash.is_zero() {
            warn!(target: "derivation", engine_safe_head = ?engine_safe_head.block_info.number, "Waiting for engine to initialize state prior to derivation.");
            return Ok(());
        }

        // Advance the pipeline as much as possible, new data may be available or there still may be
        // payloads in the attributes queue.
        let payload_attrs = match self.produce_next_attributes().await {
            Ok(attrs) => attrs,
            Err(DerivationError::Yield) => {
                // Yield until more data is available.
                self.derivation_idle = true;
                return Ok(());
            }
            Err(e) => {
                return Err(e);
            }
        };

        // Mark derivation as busy.
        self.derivation_idle = false;

        // Mark the L2 safe head as seen.
        self.engine_l2_safe_head.borrow_and_update();

        // Send payload attributes out for processing.
        self.attributes_out
            .send(payload_attrs)
            .await
            .map_err(|e| DerivationError::Sender(Box::new(e)))?;

        Ok(())
    }
}

/// Messages that the [DerivationActor] can receive from other actors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InboundDerivationMessage {
    /// New data is potentially available for processing on the data availability layer.
    NewDataAvailable,
    /// The engine has updated its safe head. An attempt to process the next payload attributes can
    /// be made.
    SafeHeadUpdated,
}

/// An error from the [DerivationActor].
#[derive(Error, Debug)]
pub enum DerivationError {
    /// An error originating from the derivation pipeline.
    #[error(transparent)]
    Pipeline(#[from] PipelineErrorKind),
    /// Waiting for more data to be available.
    #[error("Waiting for more data to be available")]
    Yield,
    /// An error originating from the broadcast sender.
    #[error("Failed to send event to broadcast sender: {0}")]
    Sender(Box<dyn std::error::Error>),
    /// An error from the signal receiver.
    #[error("Failed to receive signal")]
    SignalReceiveFailed,
    /// Unable to receive the L2 safe head to step on the pipeline.
    #[error("Failed to receive L2 safe head")]
    L2SafeHeadReceiveFailed,
}
