//! Simple thread pool for processing batches of tensor contractions.

use std::thread;
use crossbeam::channel;
use thiserror::Error;
use crate::tensor::{ Tensor, Elem, Idx };

#[derive(Debug, Error)]
pub enum PoolError {
    #[error("failed to enqueue contractions: dead thread")]
    DeadThread,

    #[error("failed to enqueue contractions: closed sender channel")]
    ClosedSenderChannel,

    #[error("failed to receive contraction result: receiver error: {0}")]
    ClosedReceiverChannel(channel::RecvError),

    #[error("encountered receiver error from within a thread: receiver error: {0}")]
    WorkerReceiverError(channel::RecvError),
}
use PoolError::*;
pub type PoolResult<T> = Result<T, PoolError>;

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum ToWorker<T, A> {
    Stop,
    Work(Tensor<T, A>, Tensor<T, A>),
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum FromWorker<T, A> {
    RecvError(channel::RecvError),
    Output(Tensor<T, A>),
}

/// A simple thread pool to contract pairs of tensors in parallel.
///
/// Workload between threads is automatically balanced by means of a
/// single-producer, multiple-consumer channel. Contracted pairs are returned in
/// the order in which the contraction operations finished. The pool as a whole
/// is meant to be reused between batches of contractions, and is **not**
/// thread-safe.
#[derive(Debug)]
pub struct ContractorPool<T, A> {
    threads: Vec<thread::JoinHandle<()>>,
    workers_in: channel::Sender<ToWorker<T, A>>,
    workers_out: channel::Receiver<FromWorker<T, A>>,
}

impl<T, A> ContractorPool<T, A>
where
    T: Idx + Send + 'static,
    A: Elem + Send + 'static,
{
    /// Create a new thread pool of `nthreads` threads.
    pub fn new(nthreads: usize) -> Self {
        let (tx_in, rx_in) = channel::unbounded();
        let (tx_out, rx_out) = channel::unbounded();
        let mut threads = Vec::with_capacity(nthreads);
        for _ in 0..nthreads {
            let worker_receiver = rx_in.clone();
            let worker_sender = tx_out.clone();
            let th = thread::spawn(move || loop {
                match worker_receiver.recv() {
                    Ok(ToWorker::Stop) => { break; },
                    Ok(ToWorker::Work(a, b)) => {
                        let c = a.multiply(b);
                        match worker_sender.send(FromWorker::Output(c)) {
                            Ok(()) => { continue; },
                            Err(err) => { panic!("sender error: {err}"); },
                        }
                    },
                    Err(err) => {
                        match worker_sender.send(FromWorker::RecvError(err)) {
                            Ok(()) => { panic!("receiver error"); },
                            Err(_) => { panic!("sender error: {err}"); },
                        }
                    },
                }
            });
            threads.push(th);
        }
        Self { threads, workers_in: tx_in, workers_out: rx_out }
    }

    /// Create a new thread pool with the number of threads equal to the number
    /// of logical CPU cores available in the current system.
    pub fn new_cpus() -> Self { Self::new(num_cpus::get()) }

    /// Create a new thread pool with the number of threads equal to the number
    /// of physical CPU cores available in the current system.
    pub fn new_physical() -> Self { Self::new(num_cpus::get_physical()) }

    /// Enqueue a batch of contractions to be distributed across all threads.
    ///
    /// This method will block until all enqueued contractions have been
    /// completed.
    pub fn do_contractions<I>(&self, pairs: I)
        -> PoolResult<Vec<Tensor<T, A>>>
    where I: IntoIterator<Item = (Tensor<T, A>, Tensor<T, A>)>
    {
        if self.threads.iter().any(|th| th.is_finished()) {
            return Err(DeadThread);
        }
        let mut count: usize = 0;
        for (t_a, t_b) in pairs.into_iter() {
            match self.workers_in.send(ToWorker::Work(t_a, t_b)) {
                Ok(()) => { count += 1; },
                Err(_) => { return Err(ClosedSenderChannel); },
            }
        }
        let mut output = Vec::with_capacity(count);
        for _ in 0..count {
            match self.workers_out.recv() {
                Ok(FromWorker::Output(t_c)) => { output.push(t_c); },
                Ok(FromWorker::RecvError(err)) => {
                    return Err(WorkerReceiverError(err));
                }
                Err(err) => { return Err(ClosedReceiverChannel(err)); },
            }
        }
        Ok(output)
    }
}

impl<T, A> Drop for ContractorPool<T, A> {
    fn drop(&mut self) {
        (0..self.threads.len())
            .for_each(|_| { self.workers_in.send(ToWorker::Stop).ok(); });
        self.threads.drain(..)
            .for_each(|th| { th.join().ok(); });
    }
}

