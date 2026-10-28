'use strict';
let queueJob = null, queueBusy = false, queueFetching = false, queueBaseline = '';
const queueStates = {running: '처리 중', pausing: '현재 호출 완료 후 중단', paused: '중단됨',
  completed: '자동 처리 완료', failed: '실패', limit_reached: '호출 한도 도달',
  waiting: '처리 대기', processing: '차팅 중', ready: '검토 대기', held: '보류',
  skipped: '건너뜀', interrupted: '결과 미확인', approved: '승인 완료'};
function queueChanged() {
  return !!queueSelection && !saved && queueBaseline !== JSON.stringify([$('records').value, $('notes').value]);
}
function queueControls() {
  reviewQueueBusy = queueBusy;
  queueActive = !!queueJob && ['running', 'pausing'].includes(queueJob.status);
  $('queue-start').disabled = queueBusy || queueActive || !csrf;
  $('queue-pause').disabled = queueBusy || !queueActive;
  $('queue-resume').disabled = queueBusy || !queueJob || queueActive;
  const item = queueJob?.items.find(row => row.id === queueSelection?.item_id);
  for (const id of ['queue-hold', 'queue-skip', 'queue-regenerate']) {
    $(id).disabled = queueBusy || !item || item.status === 'approved' || item.status === 'processing';
  }
  $('queue-regenerate').disabled ||= queueActive;
  $('queue-jobs').disabled = queueBusy;
  controls();
}
function updateQueue(job) {
  queueJob = job;
  if (queueSelection?.job_id === job.id && job.items.some(item => item.id === queueSelection.item_id && item.status === 'approved')) saved = true;
  $('queue-progress').textContent = `${queueStates[job.status] || job.status} · 요청 ${job.requested_count}건 / 생성 ${job.total}건 · ${job.processed}/${job.total}건 처리 · 남은 ${job.remaining}건 · 승인 ${job.approved}건`;
  $('queue-progress-bar').max = job.total || job.requested_count;
  $('queue-progress-bar').value = job.processed;
  const u = job.usage;
  $('queue-usage').textContent = `API 시도 ${job.calls_attempted}회 / 사용·예약 ${job.calls_reserved}/${job.call_limit}회 · 입력 ${u.input_tokens} / 출력 ${u.output_tokens} / 추론 ${u.reasoning_tokens} 토큰 (추론 미확인 ${u.unknown_reasoning_calls}회) · 사용량 미확인 ${u.unknown_calls}회 · 중복 제외 ${job.duplicates_removed}건`;
  $('queue-error').textContent = job.error || '';
  $('queue-items').replaceChildren(...job.items.map((item, index) => {
    const li = document.createElement('li'), button = document.createElement('button');
    button.type = 'button'; button.dataset.itemId = item.id;
    button.textContent = `${queueStates[item.status] || item.status} · ${item.input} (처리 시도 ${item.attempt_count}회)`;
    button.setAttribute('aria-current', String(queueSelection?.item_id === item.id));
    button.addEventListener('click', () => selectQueueItem(item.id)); li.append(button); return li;
  }));
  list('queue-calls', job.calls.map(call => `${call.purpose === 'synthetic_inputs' ? '합성 입력 생성' : '차팅'} · ${call.status} · ${call.audit?.status || '사용량/결과 미확인'}`));
  queueControls();
}
async function getQueue(url) {
  const response = await fetch(url, {credentials: 'same-origin'}), result = await response.json();
  if (!response.ok) throw new Error(result.error || '대기열을 읽지 못했습니다.');
  return result;
}
async function refreshQueues(selectId = null) {
  const result = await getQueue('/api/queue');
  const id = selectId || queueJob?.id || result.jobs[0]?.id || '';
  $('queue-jobs').replaceChildren(...(result.jobs.length ? result.jobs.map(job => {
    const option = document.createElement('option'); option.value = job.id;
    option.textContent = `${new Date(job.created_at).toLocaleString()} · ${queueStates[job.status] || job.status} · ${job.total}건`;
    return option;
  }) : [new Option('대기열 없음', '')]));
  $('queue-jobs').value = id;
  if (id) updateQueue(await getQueue('/api/queue/' + encodeURIComponent(id)));
  if (result.worker_error) $('queue-error').textContent = '로컬 처리 오류: ' + result.worker_error + '. 기존 파일을 보존했습니다.';
  queueControls();
}
async function saveQueueEdits() {
  if (!queueChanged()) return;
  const result = await post('/api/queue/edit', {...queueSelection,
    records: [$('records').value], notes: $('notes').value});
  queueBaseline = JSON.stringify([$('records').value, $('notes').value]); updateQueue(result);
}
async function selectQueueItem(identifier) {
  if (queueBusy || busy) return;
  if (draft && !queueSelection) { message('단일 입력 검토가 열려 있습니다. 새 입력 시작으로 정리한 뒤 대기열을 여세요.', true); return; }
  queueBusy = true; queueControls();
  try {
    await saveQueueEdits();
    updateQueue(await getQueue('/api/queue/' + queueJob.id));
    const item = queueJob.items.find(row => row.id === identifier);
    queueSelection = {job_id: queueJob.id, item_id: identifier};
    $('input').value = item.input;
    const view = item.view ? {...item.view, review_available: !['processing', 'waiting'].includes(item.status)} : {id: item.id, route: 'failed', review_available: false,
      classification: null, classification_status: 'unclassified', llm_attempted: queueJob.calls.some(c => c.item_id === item.id),
      records: [], uncertainties: [item.error || '아직 검토 가능한 초안이 없습니다.'], omissions: [],
      evidence_ids: [], excluded_conflicting_ids: [], gold_count: queueJob.gold_count};
    displayDraft(view, item.review?.records || [], item.review?.notes || '');
    saved = item.status === 'approved';
    $('queue-review-actions').hidden = false;
    queueBaseline = JSON.stringify([$('records').value, $('notes').value]);
    if (item.error) message(item.error, true);
    else message(`대기열: ${queueStates[item.status] || item.status}. 원문과 확인 사항을 검토하세요.`);
    updateQueue(queueJob);
  } catch (error) { message(error.message, true); }
  finally { queueBusy = false; queueControls(); }
}
async function queueAction(action) {
  if (queueBusy || busy || !queueJob) return;
  queueBusy = true; queueControls();
  try {
    await saveQueueEdits();
    const result = await post('/api/queue/' + action, {job_id: queueJob.id,
      item_id: queueSelection?.item_id, cost_ack: $('queue-cost').checked,
      call_limit: Number($('queue-limit').value)});
    updateQueue(result);
    if (action === 'regenerate') clearDraftView();
    message(action === 'pause' ? '현재 호출이 끝나면 중단합니다. 결과는 보존됩니다.' : '대기열 상태를 저장했습니다.');
  } catch (error) { message(error.message, true); }
  finally { queueBusy = false; queueControls(); }
}
$('queue-form').addEventListener('submit', async event => {
  event.preventDefault(); if (queueBusy || busy || queueActive) return;
  queueBusy = true; queueControls();
  try {
    await saveQueueEdits();
    const result = await post('/api/queue/start', {count: Number($('queue-count').value),
      call_limit: Number($('queue-limit').value), cost_ack: $('queue-cost').checked});
    updateQueue(result); await refreshQueues(result.id);
    message('합성 입력 생성과 순차 차팅을 시작했습니다. 검토 전에는 Gold에 저장하지 않습니다.');
  } catch (error) { message(error.message, true); }
  finally { queueBusy = false; queueControls(); }
});
$('queue-count').addEventListener('change', () => { $('queue-limit').value = Number($('queue-count').value) + 1; });
$('queue-jobs').addEventListener('change', async () => {
  try {
    await saveQueueEdits(); if (queueSelection) clearDraftView();
    await refreshQueues($('queue-jobs').value); $('queue-limit').value = queueJob.call_limit;
  } catch (error) { message(error.message, true); }
});
$('queue-pause').addEventListener('click', () => queueAction('pause'));
$('queue-resume').addEventListener('click', () => queueAction('resume'));
$('queue-hold').addEventListener('click', () => queueAction('hold'));
$('queue-skip').addEventListener('click', () => queueAction('skip'));
$('queue-regenerate').addEventListener('click', () => queueAction('regenerate'));
$('queue-edit').addEventListener('click', async () => {
  try { await saveQueueEdits(); message('수정 내용을 대기열에 저장했습니다. Gold에는 추가하지 않았습니다.'); }
  catch (error) { message(error.message, true); }
});
window.addEventListener('beforeunload', event => { if (queueChanged()) { event.preventDefault(); event.returnValue = ''; } });
let polls = 0;
async function pollQueue() {
  if (queueFetching || queueBusy || busy) return;
  queueFetching = true;
  try {
    await refreshQueues();
    if (++polls === 1 && queueJob) $('queue-limit').value = queueJob.call_limit;
  } catch (error) { $('queue-error').textContent = error.message; }
  finally { queueFetching = false; }
}
pollQueue(); setInterval(pollQueue, 1200);
