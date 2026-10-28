'use strict';
const $ = id => document.getElementById(id);
let csrf = '', draft = null, busy = false, saved = false;
let queueSelection = null, queueActive = false, reviewQueueBusy = false;
function message(text, error = false) {
  $('message').textContent = text;
  $('message').className = error ? 'error' : '';
}
function controls() {
  $('generate').disabled = busy || reviewQueueBusy || !!draft || !csrf || queueActive;
  $('new').disabled = busy || reviewQueueBusy;
  for (const id of ['input', 'synthetic']) $(id).disabled = busy || reviewQueueBusy || !!draft;
  const editable = !!draft && !busy && !reviewQueueBusy && !saved && draft.route !== 'conflict' && draft.review_available !== false && draft.records.length <= 1;
  for (const id of ['records', 'reviewer', 'notes', 'approved']) $(id).disabled = !editable;
  if ($('queue-edit')) $('queue-edit').disabled = !editable;
  $('save').disabled = !editable || !$('approved').checked || !$('reviewer').value.trim() || !$('records').value.trim();
}
async function post(path, data) {
  const response = await fetch(path, {method: 'POST', credentials: 'same-origin',
    headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, body: JSON.stringify(data)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '요청에 실패했습니다.');
  return result;
}
function list(id, values) {
  $(id).replaceChildren(...values.map(text => {
    const item = document.createElement('li'); item.textContent = text; return item;
  }));
}
function showClassification(result) {
  const conflict = result.route === 'conflict' || result.classification_status === 'conflict';
  const labels = {general_transform: '일반 변환', set_charting: 'Set Charting',
    conditional_restore: '조건부 복원', previous_record_reference: '이전 기록 참조', input_typo: '입력 오타'};
  const guidance = {
    general_transform: '사용자가 일반 변환으로 승인한 원문·맥락과 일치합니다. 수치·단위·부정 표현이 현재 입력과 일치하는지 검토하세요.',
    set_charting: '승인된 Set의 정확 입력에 대한 미승인 초안입니다. 복원된 상태·처치·교육이 실제 관찰·시행되었는지 원문과 대조하세요. 시행하지 않은 중재는 삭제·수정하고, 생략된 단위·경로도 확인한 뒤 승인하세요.',
    conditional_restore: '이미 조건부 복원으로 승인된 사례입니다. 생략된 상태·중재의 근거와 시행/계획 여부를 확인하기 전에는 기록을 자동 복원하지 않습니다.',
    previous_record_reference: '현재 적용할 이전 기록과 환자 상태의 근거가 필요합니다. 이전 사례의 사실을 복사하지 않고 기록 생성을 보류했습니다.',
    input_typo: '오타 검토 대상으로 분류된 입력입니다. 의도한 표현을 확인하고 새 입력으로 다시 시작하세요.'
  };
  $('classification').textContent = conflict ? 'Gold 충돌 · 분류 보류' : (labels[result.classification] || '분류 미확정');
  $('classification-help').textContent = conflict
    ? '동일 입력·맥락의 Gold 또는 학습 패턴이 상충합니다. 기존 Gold를 자동 선택·병합하지 않고 생성을 보류했습니다. 충돌 검토가 끝나기 전에는 승인·저장할 수 없습니다.'
    : (guidance[result.classification] || '승인된 원문·맥락과 정확히 일치하는 분류가 없습니다. 새 Set으로 자동 확장하지 않습니다. 생성된 문장과 임상 근거를 직접 검토하세요.');
  $('classification-help').hidden = false;
  $('route').textContent = ({exact: 'Exact', learned: 'Learned', llm: 'Luna', failed: result.llm_attempted ? 'Luna · 실패' : '생성 실패'})[result.route]
    || (result.llm_attempted ? 'Luna · 초안 보류' : '미실행 · 생성 보류');
}
async function initialise() {
  try {
    const response = await fetch('/api/state', {credentials: 'same-origin'});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    csrf = result.csrf; $('count').textContent = result.gold_count;
  } catch (error) { message(error.message, true); }
  controls();
}
function displayDraft(value, records = value.records, notes = '') {
  draft = value; saved = false;
    $('count').textContent = draft.gold_count;
    $('records').value = records.join('\n\n');
    $('notes').value = notes; $('approved').checked = false;
    showClassification(draft);
    const checks = [...draft.uncertainties];
    const notices = draft.workflow_notices || [];
    $('workflow-checks').hidden = !notices.length; list('workflow-notices', notices);
    $('gold-diagnostics').hidden = !draft.diagnostics;
    $('diagnostic-data').textContent = JSON.stringify(draft.diagnostics || {}, null, 2);
    if (draft.records.length > 1) checks.push('여러 기록의 승인은 이번 단계에서 지원하지 않습니다. 단일 기록 입력으로 다시 시작하세요.');
    $('checks').hidden = checks.length === 0; list('uncertainties', checks);
    $('details').hidden = false;
    $('reason').textContent = (draft.route === 'conflict' || draft.classification_status === 'human_check_required')
      ? $('classification-help').textContent
      : (draft.reason || (draft.route === 'llm' ? `학습 Gold ${draft.gold_sent}건을 참고했습니다.` : ''));
    if (draft.evidence_ids.length) $('reason').textContent += ' 근거 Gold: ' + draft.evidence_ids.join(', ');
    if (draft.gold_ids_sent?.length) $('reason').textContent += ' Luna에 전달한 참고 Gold (실제 임상 근거로 확정된 ID는 아님): ' + draft.gold_ids_sent.join(', ');
    list('omissions', draft.omissions);
    message(draft.route === 'conflict' ? '기존 Gold 충돌을 먼저 검토해야 합니다. 저장하지 않았습니다.'
      : (!draft.records.length ? '기록 생성을 보류했습니다. 확인 사항을 검토하고 필요한 근거를 확인하세요.'
        : '초안을 생성했습니다. 원문과 확인 사항을 검토한 뒤 승인하세요.'));
}
$('draft-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy || draft) return;
  busy = true; controls(); message('초안을 생성하고 있습니다. Luna 호출은 최대 90초가 걸릴 수 있습니다.');
  try {
    draft = await post('/api/draft', {input: $('input').value, context: '', synthetic: $('synthetic').checked});
    displayDraft(draft);
  } catch (error) { message(error.message, true); }
  finally { busy = false; controls(); }
});
$('save').addEventListener('click', async () => {
  if (busy || $('save').disabled) return;
  busy = true; controls(); message('검토한 기록을 저장하고 있습니다.');
  try {
    const review = {records: [$('records').value], reviewer: $('reviewer').value, notes: $('notes').value, approved: $('approved').checked};
    const result = queueSelection
      ? await post('/api/queue/approve', {...queueSelection, ...review})
      : await post('/api/approve', {id: draft.id, ...review});
    if (queueSelection) {
      updateQueue(result);
      const item = result.items.find(row => row.id === queueSelection.item_id);
      if (item.status !== 'approved') throw new Error(item.error || 'Gold에 저장하지 않았습니다.');
    }
    saved = true; $('count').textContent = result.gold_count;
    message('승인한 기록을 Gold에 저장했습니다. 새 입력부터 활용됩니다.');
  } catch (error) { message(error.message, true); }
  finally { busy = false; controls(); }
});
function clearDraftView() {
    draft = null; saved = false; queueSelection = null;
    if ($('queue-review-actions')) $('queue-review-actions').hidden = true;
    for (const id of ['input', 'records', 'notes']) $(id).value = '';
    $('synthetic').checked = false; $('approved').checked = false;
    $('checks').hidden = true; $('workflow-checks').hidden = true; $('gold-diagnostics').hidden = true; $('details').hidden = true; $('route').textContent = '초안 대기';
    $('classification').textContent = '분류 대기'; $('classification-help').textContent = ''; $('classification-help').hidden = true;
}
$('new').addEventListener('click', async () => {
  if (busy) return;
  busy = true; controls();
  try {
    if (queueSelection) await saveQueueEdits();
    if (draft && !queueSelection) await post('/api/discard', {id: draft.id});
    clearDraftView();
    message(''); await initialise();
  } catch (error) { message(error.message, true); }
  finally { busy = false; controls(); $('input').focus(); }
});
for (const id of ['records', 'reviewer', 'approved']) $(id).addEventListener('input', controls);
initialise();
