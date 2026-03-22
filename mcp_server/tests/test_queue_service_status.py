import asyncio

import pytest

from graphiti_core.llm_client.errors import RateLimitError
from services.queue_service import QueueService


class _BlockingGraphitiClient:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def add_episode(self, **kwargs):
        self.started.set()
        await self.release.wait()


class _FailingGraphitiClient:
    def __init__(self):
        self.started = asyncio.Event()

    async def add_episode(self, **kwargs):
        self.started.set()
        raise RuntimeError('boom')


class _RateLimitedGraphitiClient:
    def __init__(self, failures_before_success: int = 1):
        self.started = asyncio.Event()
        self.attempts = 0
        self.failures_before_success = failures_before_success

    async def add_episode(self, **kwargs):
        self.started.set()
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise RateLimitError('Rate limit exceeded')


async def _wait_for(predicate, timeout: float = 1.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError('condition not met before timeout')


@pytest.mark.asyncio
async def test_queue_service_tracks_successful_episode_lifecycle():
    queue_service = QueueService()
    graphiti_client = _BlockingGraphitiClient()
    await queue_service.initialize(graphiti_client)

    queue_position = await queue_service.add_episode(
        group_id='group-1',
        name='Episode',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-1',
    )

    assert queue_position == 1
    queued_status = queue_service.get_episode_status('episode-1')
    assert queued_status is not None
    assert queued_status.state == 'queued'

    await asyncio.wait_for(graphiti_client.started.wait(), timeout=1.0)

    processing_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-1')
        if queue_service.get_episode_status('episode-1').state == 'processing'
        else None
    )
    assert processing_status.started_at is not None
    assert processing_status.processed_at is None

    graphiti_client.release.set()
    completed_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-1')
        if queue_service.get_episode_status('episode-1').state == 'completed'
        else None
    )
    assert completed_status.processed_at is not None
    assert completed_status.last_error is None


@pytest.mark.asyncio
async def test_queue_service_tracks_failed_episode_lifecycle():
    queue_service = QueueService()
    graphiti_client = _FailingGraphitiClient()
    await queue_service.initialize(graphiti_client)

    await queue_service.add_episode(
        group_id='group-1',
        name='Episode',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-2',
    )

    await asyncio.wait_for(graphiti_client.started.wait(), timeout=1.0)
    failed_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-2')
        if queue_service.get_episode_status('episode-2').state == 'failed'
        else None
    )
    assert failed_status.last_error == 'boom'
    assert failed_status.processed_at is not None


@pytest.mark.asyncio
async def test_queue_service_reports_pending_queue_position():
    queue_service = QueueService()
    graphiti_client = _BlockingGraphitiClient()
    await queue_service.initialize(graphiti_client)

    await queue_service.add_episode(
        group_id='group-1',
        name='Episode 1',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-3',
    )
    await asyncio.wait_for(graphiti_client.started.wait(), timeout=1.0)

    queue_position = await queue_service.add_episode(
        group_id='group-1',
        name='Episode 2',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-4',
    )

    assert queue_position == 1
    pending_status = queue_service.get_episode_status('episode-4')
    assert pending_status is not None
    assert pending_status.state == 'queued'
    assert queue_service.get_queue_position('episode-4') == 1


@pytest.mark.asyncio
async def test_queue_service_retries_rate_limited_episode_until_completed(monkeypatch):
    queue_service = QueueService(
        max_retries=2,
        retry_base_delay_seconds=0.01,
        retry_jitter_seconds=0.0,
        rate_limit_cooldown_base_seconds=0.0,
        rate_limit_cooldown_max_seconds=0.0,
        rate_limit_cooldown_jitter_seconds=0.0,
    )
    graphiti_client = _RateLimitedGraphitiClient(failures_before_success=1)
    await queue_service.initialize(graphiti_client)

    real_sleep = asyncio.sleep

    async def _no_sleep(_: float):
        await real_sleep(0)

    monkeypatch.setattr('services.queue_service.asyncio.sleep', _no_sleep)

    await queue_service.add_episode(
        group_id='group-1',
        name='Episode',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-5',
    )

    await asyncio.wait_for(graphiti_client.started.wait(), timeout=1.0)
    completed_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-5')
        if queue_service.get_episode_status('episode-5').state == 'completed'
        else None
    )

    assert graphiti_client.attempts == 2
    assert completed_status.attempt_count == 2
    assert completed_status.last_error_code == 'rate_limit'
    assert completed_status.next_retry_at is None
    assert completed_status.processed_at is not None


@pytest.mark.asyncio
async def test_queue_service_marks_rate_limited_episode_failed_after_retry_budget(monkeypatch):
    queue_service = QueueService(
        max_retries=1,
        retry_base_delay_seconds=0.01,
        retry_jitter_seconds=0.0,
        rate_limit_cooldown_base_seconds=0.0,
        rate_limit_cooldown_max_seconds=0.0,
        rate_limit_cooldown_jitter_seconds=0.0,
    )
    graphiti_client = _RateLimitedGraphitiClient(failures_before_success=10)
    await queue_service.initialize(graphiti_client)

    real_sleep = asyncio.sleep

    async def _no_sleep(_: float):
        await real_sleep(0)

    monkeypatch.setattr('services.queue_service.asyncio.sleep', _no_sleep)

    await queue_service.add_episode(
        group_id='group-1',
        name='Episode',
        content='content',
        source_description='test',
        episode_type='text',
        entity_types=None,
        uuid='episode-6',
    )

    failed_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-6')
        if queue_service.get_episode_status('episode-6').state == 'failed'
        else None
    )

    assert graphiti_client.attempts == 2
    assert failed_status.attempt_count == 2
    assert failed_status.last_error_code == 'rate_limit'
    assert failed_status.processed_at is not None


@pytest.mark.asyncio
async def test_queue_service_applies_global_cooldown_across_groups(monkeypatch):
    queue_service = QueueService(
        max_retries=0,
        retry_jitter_seconds=0.0,
        rate_limit_cooldown_base_seconds=0.01,
        rate_limit_cooldown_max_seconds=0.01,
        rate_limit_cooldown_jitter_seconds=0.0,
    )

    sleep_calls = []
    group_b_processed = asyncio.Event()
    first_group_attempted = asyncio.Event()
    real_sleep = asyncio.sleep

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)
        await real_sleep(delay)

    monkeypatch.setattr('services.queue_service.asyncio.sleep', _fake_sleep)

    async def _rate_limited_once():
        first_group_attempted.set()
        raise RateLimitError('HTTP 429 Too Many Requests')

    async def _group_b_process():
        group_b_processed.set()

    await queue_service.add_episode_task('group-a', 'episode-a', _rate_limited_once)
    await asyncio.wait_for(first_group_attempted.wait(), timeout=1.0)

    failed_status = await _wait_for(
        lambda: queue_service.get_episode_status('episode-a')
        if queue_service.get_episode_status('episode-a').state == 'failed'
        else None
    )
    assert failed_status.last_error_code == 'rate_limit'

    await queue_service.add_episode_task('group-b', 'episode-b', _group_b_process)
    await asyncio.wait_for(group_b_processed.wait(), timeout=1.0)

    assert any(delay > 0 for delay in sleep_calls)
