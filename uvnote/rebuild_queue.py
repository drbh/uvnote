"""Rebuild queue manager with debouncing and cancellation support."""

import threading
import time
import queue
from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path
import subprocess

from .logging_config import get_logger

logger = get_logger("rebuild_queue")


@dataclass
class RebuildRequest:
    """A rebuild request with timestamp."""

    timestamp: float
    file_path: Path
    request_id: int


class RebuildQueueManager:
    """Manages rebuild requests with debouncing and cancellation."""

    def __init__(
        self,
        rebuild_func: Callable[[Optional[threading.Event]], None],
        loading_func: Optional[Callable[[], None]] = None,
        debounce_seconds: float = 0.5,
        min_interval_seconds: float = 1.0,
    ):
        """
        Initialize the rebuild queue manager.

        Args:
            rebuild_func: The function to call for rebuilding (takes cancel_event)
            loading_func: Optional function to call immediately when rebuild is requested
            debounce_seconds: Time to wait after last change before rebuilding
            min_interval_seconds: Minimum time between rebuild starts
        """
        self.rebuild_func = rebuild_func
        self.loading_func = loading_func
        self.debounce_seconds = debounce_seconds
        self.min_interval_seconds = min_interval_seconds

        self._queue: queue.Queue[Optional[RebuildRequest]] = queue.Queue()
        self._stop_event = threading.Event()
        self._current_cancel_event: Optional[threading.Event] = None
        self._current_thread: Optional[threading.Thread] = None
        self._last_rebuild_start = 0.0
        self._request_counter = 0
        self._lock = threading.Lock()

        # Start the worker thread
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        logger.info(
            f"RebuildQueueManager initialized with debounce={debounce_seconds}s, min_interval={min_interval_seconds}s"
        )

    def request_rebuild(self, file_path: Path) -> None:
        """Request a rebuild for the given file."""
        with self._lock:
            self._request_counter += 1
            request_id = self._request_counter

        request = RebuildRequest(
            timestamp=time.time(), file_path=file_path, request_id=request_id
        )

        logger.info(f"Rebuild requested: id={request_id} file={file_path.name}")

        # Immediately emit loading state
        if self.loading_func:
            try:
                logger.info(f"Emitting loading state for request id={request_id}")
                self.loading_func()
            except Exception as e:
                logger.error(f"Failed to emit loading state: {e}", exc_info=True)

        self._queue.put(request)

    def stop(self) -> None:
        """Stop the queue manager and cancel any running rebuilds."""
        logger.info("Stopping RebuildQueueManager")
        self._stop_event.set()
        self._queue.put(None)  # Wake up the worker
        self._cancel_current_rebuild()

        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def _worker(self) -> None:
        """Worker thread that processes rebuild requests."""
        logger.debug("Worker thread started")
        pending_request: Optional[RebuildRequest] = None

        while not self._stop_event.is_set():
            try:
                # Wait for new requests or timeout for debouncing
                timeout = self.debounce_seconds if pending_request else None

                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    # Timeout reached, process pending request if any
                    if pending_request:
                        self._process_rebuild(pending_request)
                        pending_request = None
                    continue

                if item is None:  # Stop signal
                    break

                # New request received
                if pending_request:
                    logger.info(
                        f"Dropping pending request id={pending_request.request_id} in favor of newer id={item.request_id}"
                    )

                pending_request = item

                # Drain queue to get the latest request
                dropped_count = 0
                while True:
                    try:
                        newer_item = self._queue.get_nowait()
                        if newer_item is None:  # Stop signal
                            return
                        dropped_count += 1
                        pending_request = newer_item
                    except queue.Empty:
                        break

                if dropped_count > 0:
                    logger.info(
                        f"Dropped {dropped_count} intermediate requests, using latest id={pending_request.request_id}"
                    )

            except Exception as e:
                logger.error(f"Worker thread error: {e}", exc_info=True)

        logger.debug("Worker thread stopped")

    def _process_rebuild(self, request: RebuildRequest) -> None:
        """Process a rebuild request."""
        # Check minimum interval
        time_since_last = time.time() - self._last_rebuild_start
        if time_since_last < self.min_interval_seconds:
            wait_time = self.min_interval_seconds - time_since_last
            logger.info(f"Waiting {wait_time:.2f}s before rebuild (min interval)")
            time.sleep(wait_time)

        # Cancel any running rebuild
        self._cancel_current_rebuild()

        # Start new rebuild
        logger.info(f"Starting rebuild for request id={request.request_id}")
        self._last_rebuild_start = time.time()

        # Run rebuild in a separate thread so we can cancel it
        rebuild_thread = threading.Thread(
            target=self._run_rebuild, args=(request,), daemon=True
        )

        with self._lock:
            self._current_thread = rebuild_thread

        rebuild_thread.start()

    def _run_rebuild(self, request: RebuildRequest) -> None:
        """Run the rebuild function."""
        cancel_event = threading.Event()

        with self._lock:
            self._current_cancel_event = cancel_event

        start_time = time.time()
        try:
            logger.info(f"Executing rebuild for request id={request.request_id}")
            self.rebuild_func(cancel_event)
            duration = time.time() - start_time
            if cancel_event.is_set():
                logger.warning(
                    f"Rebuild cancelled for request id={request.request_id} after {duration:.2f}s"
                )
            else:
                logger.info(
                    f"Rebuild completed for request id={request.request_id} in {duration:.2f}s"
                )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Rebuild failed for request id={request.request_id} after {duration:.2f}s: {e}",
                exc_info=True,
            )
        finally:
            with self._lock:
                self._current_thread = None
                self._current_cancel_event = None

    def _cancel_current_rebuild(self) -> None:
        """Cancel the currently running rebuild if any."""
        with self._lock:
            if self._current_thread and self._current_thread.is_alive():
                logger.warning("Cancelling running rebuild")
                if self._current_cancel_event:
                    self._current_cancel_event.set()
                else:
                    logger.warning("No cancel event available for current rebuild")
