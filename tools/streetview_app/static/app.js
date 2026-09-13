'use strict';

(() => {
  const $ = id => document.getElementById(id);
  const state = {center: null, coordinateDraft: false, radius: 100, snapshot: null, continuation: null, discoveryDate: null, panoramas: [], dateOptions: [], filtered: [], selected: new Set(), manualIncluded: new Set(), excluded: new Set(), mode: 'same_day', active: null, status: null, discovering: false, submitting: false, request: 0, previewRequest: 0};
  const markers = new Map();
  let map = null, scopeCircle = null, centerMarker = null, toastTimer = null, statusBusy = false;
  const filterState = {ready: false, submitting: false, defaultsApplied: false, touched: false, jobs: []};
  const depthState = {defaultApplied: false};
  const faceNames = {F: '앞', R: '오른쪽', B: '뒤', L: '왼쪽', U: '위', D: '아래'};

  function text(element, value) { element.textContent = value == null ? '' : String(value); }
  function el(tag, className, value) { const node = document.createElement(tag); if (className) node.className = className; if (value != null) text(node, value); return node; }
  function toast(message, error = false) {
    text($('toast'), message); $('toast').classList.toggle('error', error); $('toast').hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 8000 : 4500);
  }
  async function api(path, options = {}) {
    const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), options.timeout || 90000);
    try {
      const response = await fetch(path, {cache: 'no-store', signal: controller.signal, ...options, headers: {'Content-Type': 'application/json', ...(options.headers || {})}});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `요청을 완료하지 못했습니다 (${response.status}).`);
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.');
      throw error;
    } finally { clearTimeout(timeout); }
  }
  const validCenter = point => point && Number.isFinite(point.lat) && Number.isFinite(point.lng) && Math.abs(point.lat) <= 90 && Math.abs(point.lng) <= 180;
  const validRadius = () => Number.isFinite(state.radius) && state.radius >= 10;
  function syncRadiusSlider() {
    if (!validRadius()) return;
    const maximum = Math.max(1000, state.radius);
    $('radius').max = String(maximum);
    $('radius').value = state.radius;
    text($('radius-range-max'), `${maximum.toLocaleString()} m`);
  }
  function scopeDirty() { return state.coordinateDraft || (!!state.snapshot && (!validCenter(state.center) || state.center.lat !== state.snapshot.center.lat || state.center.lng !== state.snapshot.center.lng || state.radius !== state.snapshot.radius_m)); }
  const inputIds = () => [...state.selected];
  const validSizePercent = value => value !== '' && Number.isFinite(Number(value)) && Number(value) >= .01 && Number(value) <= 1000;
  function generationSizeOptions() {
    const enabled = $('size-filter-enabled').checked;
    const value = $('size-filter-percent').value;
    if (enabled && !validSizePercent(value)) throw new Error('크기 제한은 0.01~1,000%로 입력해 주세요.');
    return {enabled, max_sigma_camera_radius_ratio: validSizePercent(value) ? Number(value) / 100 : .5};
  }
  function depthCleanupCapability() {
    return window.streetviewWorkflow?.depthCleanupCapability?.() || state.status?.depth_cleanup || {available:false};
  }
  function generationDepthOptions() {
    return depthCleanupCapability().available ? {enabled:$('depth-cleanup-enabled').checked} : undefined;
  }
  function canGenerate() {
    const modes = state.status?.capabilities?.generation_modes;
    return typeof modes?.multi_view === 'boolean' ? modes.multi_view : state.status?.capabilities?.generate === true;
  }
  function updateControls() {
    const dirty = scopeDirty();
    $('discover').disabled = !validCenter(state.center) || state.coordinateDraft || !validRadius() || state.discovering || state.submitting;
    $('discover-label').textContent = state.discovering ? '촬영 자료를 조회하는 중…' : dirty ? '변경한 범위 다시 조회' : '주변 거리뷰 조회';
    $('discover').classList.toggle('loading', state.discovering);
    $('fit-area').disabled = !validCenter(state.center);
    $('scope-note').classList.toggle('warning', dirty);
    text($('scope-note'), state.coordinateDraft ? '입력한 좌표를 이동 버튼(↗)으로 적용해 주세요.' : dirty ? '범위가 변경되었습니다. 저장 전에 다시 조회해 주세요.' : !state.center ? '중심 위치를 선택하면 조회할 수 있습니다.' : !validRadius() ? '반경은 10 m 이상으로 입력해 주세요.' : state.snapshot ? `조회 기준: ${state.snapshot.center.lat.toFixed(5)}, ${state.snapshot.center.lng.toFixed(5)} · ${state.snapshot.radius_m.toLocaleString()} m` : '주변 거리뷰 조회를 누르면 실제 촬영 자료를 확인합니다.');
    const count = inputIds().length;
    const sizeReady = !$('size-filter-enabled').checked || validSizePercent($('size-filter-percent').value);
    const workflowReady = window.streetviewWorkflow?.generationReady?.() !== false;
    const ready = !!state.snapshot && !dirty && !state.discovering && !state.submitting && count > 0 && validPolicy() && sizeReady;
    $('save-plan').disabled = !ready;
    $('generate').disabled = !ready || !canGenerate() || !workflowReady;
    text($('selected-count'), count.toLocaleString());
    text($('excluded-count'), state.filtered.filter(p => state.excluded.has(p.id)).length.toLocaleString());
    $('select-all').disabled = !state.filtered.length || state.discovering || state.submitting || dirty;
    text($('select-all'), '전체 포함');
    $('clear-selection').disabled = !state.selected.size || state.submitting || state.discovering || dirty;
    $('station-list').querySelectorAll('input').forEach(input => { input.disabled = dirty || state.discovering || state.submitting; });
    $('rescan-date').disabled = !state.snapshot || dirty || state.discovering || state.submitting || !currentPolicy().value;
    $('load-more').hidden = !state.continuation;
    $('load-more').disabled = dirty || state.discovering || state.submitting || (state.discoveryDate !== null && state.discoveryDate !== currentPolicy().value);
    text($('load-more'), state.discovering ? '불러오는 중…' : '거리뷰 더 불러오기');
    document.querySelectorAll('[data-mode]').forEach(button => { button.disabled = state.discovering || state.submitting; });
    $('capture-date').disabled = state.discovering || state.submitting || !state.snapshot || state.mode === 'any' || !$('capture-date').value;
    $('remove-sky').disabled = state.submitting; $('mask-dynamic').disabled = state.submitting;
    $('size-filter-enabled').disabled = state.submitting;
    $('size-filter-percent').disabled = state.submitting || !$('size-filter-enabled').checked;
    $('size-filter-percent').setAttribute('aria-invalid', String(!sizeReady));
    const depthCapability = depthCleanupCapability();
    if (!depthState.defaultApplied && depthCapability.available) {
      $('depth-cleanup-enabled').checked = depthCapability.default_enabled !== false;
      depthState.defaultApplied = true;
    }
    $('depth-cleanup-enabled').disabled = state.submitting || !depthCapability.available;
    text($('depth-cleanup-note'), depthCapability.available ? '깊이·하늘 정보를 참고해 부유물을 정리합니다. 바닥 보호를 유지합니다.' : '뎁스 정리가 연결된 제작 프리셋에서 사용할 수 있습니다.');
    const disabledReason = state.status?.disabled_reasons?.multi_view || state.status?.disabled_reasons?.generate;
    text($('generation-note'), state.status == null ? '서버의 생성 엔진 상태를 확인하고 있습니다.' : canGenerate() ? (ready ? '선택한 실제 촬영 자료로 생성 작업을 시작합니다.' : '촬영 지점을 선택하면 생성을 시작할 수 있습니다.') : disabledReason || '생성 엔진이 아직 연결되지 않았습니다. 선택한 설정은 저장할 수 있습니다.');
    const current = state.active && state.panoramas.find(p => p.id === state.active.id);
    $('toggle-preview-selection').disabled = !current || !state.filtered.some(p => p.id === current.id) || dirty || state.discovering || state.submitting;
    text($('toggle-preview-selection'), current && state.selected.has(current.id) ? '학습에서 제외' : '학습에 포함');
    $('toggle-preview-selection').classList.toggle('exclude-action', !!current && state.selected.has(current.id));
    updateNaverLink();
    if (!workflowReady) text($('generation-note'), window.streetviewWorkflow?.generationReason?.() || '선택한 프리셋의 연결 상태를 확인해 주세요.');
    window.streetviewWorkflow?.lockRecipeControls?.();
  }
  function updateNaverLink() {
    const link = $('naver-roadview'), pano = state.active;
    if (!pano || typeof pano.id !== 'string' || !pano.id) {
      link.removeAttribute('href'); link.setAttribute('aria-disabled', 'true'); link.tabIndex = -1; return;
    }
    // Naver's web panorama route takes an exact capture ID, heading, tilt, FOV,
    // and viewer mode. The ID keeps historical imagery tied to this selection.
    const url = new URL('https://map.naver.com/p/');
    const heading = Number.isFinite(Number(pano.heading)) ? Math.round(Number(pano.heading)) : 0;
    url.searchParams.set('p', [pano.id, heading, 0, 80, 'Float'].join(','));
    link.href = url.href; link.setAttribute('aria-disabled', 'false'); link.tabIndex = 0;
  }
  function setCenter(lat, lng, fit = false) {
    const point = {lat: Number(lat), lng: Number(lng)};
    if (!validCenter(point)) { toast('올바른 위도와 경도를 입력해 주세요.', true); return; }
    state.center = point; state.coordinateDraft = false;
    $('latitude').value = point.lat.toFixed(7); $('longitude').value = point.lng.toFixed(7);
    $('map-prompt').hidden = true;
    drawScope(fit); updateControls();
  }
  function drawScope(fit = false) {
    if (!state.center) return;
    text($('map-radius-label'), `${validRadius() ? state.radius.toLocaleString() : '—'} m 반경`);
    if (!map) return;
    const latlng = [state.center.lat, state.center.lng];
    if (!centerMarker) centerMarker = L.marker(latlng, {interactive: false, icon: L.divIcon({className: 'center-marker', html: '⌖', iconSize: [26, 26], iconAnchor: [13, 13]})}).addTo(map);
    else centerMarker.setLatLng(latlng);
    if (validRadius()) {
      if (!scopeCircle) scopeCircle = L.circle(latlng, {radius: state.radius, color: '#c3edd0', weight: 1.3, fillColor: '#b2dbbc', fillOpacity: .10, dashArray: '5 5', interactive: false}).addTo(map);
      else scopeCircle.setLatLng(latlng).setRadius(state.radius);
      if (fit) map.fitBounds(scopeCircle.getBounds(), {padding: [45, 45], maxZoom: 19});
    }
  }
  function setupMap() {
    if (!window.L) {
      const box = el('div', 'map-unavailable'); box.append(el('p', '', '지도를 불러오지 못했습니다. 인터넷 연결을 확인하거나 왼쪽 좌표 입력으로 범위를 지정해 주세요.'));
      $('map').append(box); $('map-prompt').hidden = true; return;
    }
    map = L.map('map', {preferCanvas: true, zoomControl: false, attributionControl: true, minZoom: 3}).setView([36.4, 127.8], 7);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors'}).addTo(map);
    L.control.zoom({position: 'topright'}).addTo(map);
    map.on('click', event => { if (!state.discovering) setCenter(event.latlng.lat, event.latlng.lng); });
    new ResizeObserver(() => map.invalidateSize()).observe($('map'));
  }
  function captureParts(pano) {
    const precision = pano.capture_precision || 'unknown';
    const raw = String(pano.captured_at || pano.capture_date || '');
    const match = /^(\d{4}-\d{2})(?:-(\d{2}))?(?:[T ](\d{2}:\d{2}))?/.exec(raw);
    if (!match || precision === 'unknown') return {month: null, day: null, clock: null};
    return {month: match[1], day: match[2] && !['month', 'year'].includes(precision) ? `${match[1]}-${match[2]}` : null, clock: match[3] && !['month', 'year', 'day', 'date'].includes(precision) ? match[3] : null};
  }
  function captureLabel(pano) {
    const parts = captureParts(pano);
    if (parts.day) return `${parts.day}${parts.clock ? ` · ${parts.clock}` : ''}`;
    if (parts.month) return `${parts.month} · 월 단위 제공`;
    return '촬영 날짜 미확인';
  }
  function currentPolicy() { return {mode: state.mode, value: state.mode === 'any' ? null : $('capture-date').value || null}; }
  function validPolicy() {
    const p = currentPolicy();
    return p.mode === 'any' || !!p.value;
  }
  function populateDates(preferred = null) {
    const values = new Map();
    const sources = [...state.dateOptions.map(d => ({value: d.value, precision: d.precision, count: d.count || 0})), ...state.panoramas.map(p => ({value: p.capture_date || String(p.captured_at || '').slice(0, 10), precision: p.capture_precision, count: 0}))];
    for (const item of sources) {
      const raw = String(item.value || '');
      let value = null;
      if (state.mode === 'same_month' && /^\d{4}-\d{2}(?:-\d{2})?$/.test(raw)) value = raw.slice(0, 7);
      if (state.mode === 'same_day' && /^\d{4}-\d{2}-\d{2}$/.test(raw) && !['year', 'month', 'unknown'].includes(item.precision)) value = raw;
      if (value) values.set(value, Math.max(values.get(value) || 0, item.count));
    }
    const select = $('capture-date'); select.replaceChildren();
    if (state.mode === 'any') {
      const option = el('option', '', '확인된 모든 촬영 시기'); option.value = ''; select.append(option); select.disabled = true;
    } else if (!values.size) {
      const option = el('option', '', state.snapshot ? (state.mode === 'same_day' ? '정확한 촬영 날짜가 없습니다' : '확인된 촬영 월이 없습니다') : '먼저 거리뷰를 조회하세요'); option.value = ''; select.append(option); select.disabled = true;
    } else {
      for (const value of [...values.keys()].sort().reverse()) {
        const present = state.panoramas.filter(p => state.mode === 'same_day' ? captureParts(p).day === value : captureParts(p).month === value).length;
        const option = el('option', '', `${value}${state.mode === 'same_month' ? ' · 월 전체' : ''}${present ? ` · 현재 ${present}개 지점` : ' · 이 날짜로 조회 가능'}`); option.value = value; select.append(option);
      }
      if (preferred && values.has(preferred)) select.value = preferred;
      select.disabled = state.discovering;
    }
    text($('date-count'), state.snapshot ? `${values.size}개 ${state.mode === 'same_month' ? '월' : '날짜'}` : '조회 후 선택');
  }
  function applyFilters(notify = false) {
    const policy = currentPolicy();
    state.filtered = validPolicy() ? state.panoramas.filter(p => {
      const parts = captureParts(p);
      if (policy.mode === 'same_day' && parts.day !== policy.value) return false;
      if (policy.mode === 'same_month' && parts.month !== policy.value) return false;
      return true;
    }) : [];
    const eligible = state.filtered.filter(p => !state.excluded.has(p.id)).map(p => p.id);
    const eligibleIds = new Set(eligible);
    const priority = [...state.manualIncluded].filter(id => eligibleIds.has(id));
    const retained = [...state.selected].filter(id => eligibleIds.has(id));
    state.selected = new Set([...priority, ...retained, ...eligible]);
    $('saved-link').hidden = true;
    if (notify) toast(`현재 촬영 조건: 학습 포함 ${state.selected.size}개 · 제외 ${state.filtered.filter(p => state.excluded.has(p.id)).length}개`);
    renderStations(); renderMarkers(); updateControls();
  }
  async function discover(dateRescan = false, append = false) {
    if (state.discovering || state.submitting || state.coordinateDraft || !validCenter(state.center) || !validRadius() || ((dateRescan || append) && (!state.snapshot || scopeDirty())) || (append && !state.continuation)) return;
    const preferred = currentPolicy().value;
    const snapshot = dateRescan || append ? structuredClone(state.snapshot) : {center: {...state.center}, radius_m: state.radius};
    const queryDate = append ? state.discoveryDate : dateRescan ? preferred : null;
    const number = ++state.request;
    state.discovering = true; updateControls();
    text($('map-status'), '실제 촬영 자료 조회 중…');
    const params = new URLSearchParams({lat: snapshot.center.lat, lng: snapshot.center.lng, radius_m: snapshot.radius_m, max_nodes: '200'});
    // max_nodes is a page size; every returned continuation may add more input.
    if (queryDate) params.set('date', queryDate);
    if (append) params.set('continuation', state.continuation);
    try {
      const data = await api(`/api/discover?${params}`);
      if (number !== state.request) return;
      if (!Array.isArray(data.panoramas)) throw new Error('촬영 지점 응답의 형식이 올바르지 않습니다.');
      const sameScope = state.snapshot && state.snapshot.center.lat === snapshot.center.lat && state.snapshot.center.lng === snapshot.center.lng && state.snapshot.radius_m === snapshot.radius_m;
      if (!sameScope) { state.excluded.clear(); state.manualIncluded.clear(); state.selected.clear(); }
      state.snapshot = snapshot;
      const seen = new Set();
      const previousCount = append ? state.panoramas.length : 0;
      state.panoramas = [...(append ? state.panoramas : []), ...data.panoramas].filter(p => {
        if (typeof p.id !== 'string' || !p.id || seen.has(p.id)) return false;
        const lng = Number(p.lng ?? p.lon); if (!validCenter({lat: Number(p.lat), lng})) return false;
        seen.add(p.id); p.lat = Number(p.lat); p.lng = lng; return true;
      });
      state.continuation = typeof data.continuation === 'string' && data.continuation ? data.continuation : null;
      state.discoveryDate = queryDate;
      state.dateOptions = Array.isArray(data.date_options) ? data.date_options : [];
      const warnings = (Array.isArray(data.warnings) ? data.warnings : []).map(String);
      if (data.truncated && !state.continuation) warnings.unshift('일부 자료만 확인했습니다. 다시 조회해 주세요.');
      text($('warning-banner'), [...new Set(warnings)].join(' ')); $('warning-banner').hidden = !warnings.length;
      if (!dateRescan && !append && state.mode === 'same_day' && !state.panoramas.some(p => captureParts(p).day) && state.panoramas.some(p => captureParts(p).month)) state.mode = 'same_month';
      syncModeButtons(); populateDates(dateRescan || append ? preferred : null); applyFilters();
      text($('map-status'), `${state.panoramas.length.toLocaleString()}개 확인${data.truncated ? ' · 일부 결과' : ''}`);
      toast(append ? `${state.panoramas.length - previousCount}개 추가 · 총 ${state.panoramas.length.toLocaleString()}개 확인` : state.panoramas.length ? `${state.panoramas.length}개 촬영 지점을 확인했습니다.` : '이 범위에서 조건에 맞는 거리뷰를 찾지 못했습니다.');
    } catch (error) {
      text($('map-status'), '조회 실패 · 다시 시도해 주세요'); toast(error.message, true);
    } finally { state.discovering = false; $('capture-date').disabled = !state.snapshot || state.mode === 'any' || !$('capture-date').value; updateControls(); }
  }
  function syncModeButtons() { document.querySelectorAll('[data-mode]').forEach(button => { const active = button.dataset.mode === state.mode; button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active)); }); }
  function toggleSelection(id) {
    if (scopeDirty() || state.discovering || state.submitting || !state.filtered.some(p => p.id === id)) return;
    if (state.selected.has(id)) { state.selected.delete(id); state.manualIncluded.delete(id); state.excluded.add(id); }
    else { state.selected.add(id); state.manualIncluded.add(id); state.excluded.delete(id); }
    $('saved-link').hidden = true; renderStations(); renderMarkers(); updateControls();
  }
  function renderStations() {
    text($('station-count'), state.filtered.length.toLocaleString());
    const list = $('station-list'); const scrollTop = list.scrollTop; list.replaceChildren();
    if (!state.filtered.length) {
      const empty = el('div', 'empty-list'); empty.append(el('span', '', '⌖'), el('p', '', state.snapshot ? (!validPolicy() ? '촬영 날짜 조건을 확인하세요.' : '현재 조건에 맞는 지점이 없습니다.\n다른 날짜로 다시 조회해 보세요.') : '선택한 범위의 거리뷰가\n여기에 표시됩니다.')); list.append(empty); return;
    }
    const fragment = document.createDocumentFragment();
    state.filtered.forEach((pano, index) => {
      const usage = state.selected.has(pano.id) ? 'selected' : state.excluded.has(pano.id) ? 'excluded' : 'unselected';
      const row = el('div', `station-row ${usage}${state.active?.id === pano.id ? ' active' : ''}`);
      row.dataset.panoramaId = pano.id;
      const checkbox = el('input'); checkbox.type = 'checkbox'; checkbox.name = 'panorama-input'; checkbox.checked = usage === 'selected'; checkbox.disabled = scopeDirty() || state.discovering || state.submitting;
      checkbox.setAttribute('aria-label', `${pano.title || `촬영 지점 ${index + 1}`} 학습 포함`); checkbox.title = '체크를 해제하면 이 지점은 생성 입력에서 제외됩니다.'; checkbox.addEventListener('change', () => toggleSelection(pano.id));
      const button = el('button', 'station-open'); button.type = 'button';
      const copy = el('span', 'station-copy'); copy.append(el('strong', '', pano.title || pano.description || `촬영 지점 ${String(index + 1).padStart(2, '0')}`), el('small', '', captureLabel(pano)));
      copy.append(el('span', 'station-use', usage === 'selected' ? '학습 포함' : usage === 'excluded' ? '학습 제외' : '미선택'));
      button.append(el('span', 'station-number', String(index + 1).padStart(2, '0')), copy, el('span', 'station-arrow', '↗'));
      button.addEventListener('click', () => openPanorama(pano));
      row.append(checkbox, button); fragment.append(row);
    });
    list.append(fragment); list.scrollTop = scrollTop;
  }
  function renderMarkers() {
    if (!map) return;
    for (const marker of markers.values()) marker.remove(); markers.clear();
    state.filtered.forEach((pano, index) => {
      const active = state.active?.id === pano.id, selected = state.selected.has(pano.id), excluded = state.excluded.has(pano.id);
      const marker = L.circleMarker([pano.lat, pano.lng], {radius: active ? 8 : selected || excluded ? 6 : 4, color: active ? '#ffffff' : selected ? '#d8f9de' : excluded ? '#edcd93' : '#a2b8ae', weight: active ? 2 : 1.5, fillColor: selected ? '#baf3cc' : excluded ? '#66513a' : '#426763', fillOpacity: selected ? 1 : .55, dashArray: excluded ? '3 2' : null});
      const tooltip = el('span', '', `${index + 1}. ${pano.title || '촬영 지점'} · ${captureLabel(pano)} · ${selected ? '학습 포함' : excluded ? '학습 제외' : '미선택'}`);
      marker.bindTooltip(tooltip, {direction: 'top'}).on('click', event => { L.DomEvent.stopPropagation(event); openPanorama(pano); }).addTo(map);
      markers.set(pano.id, marker);
    });
  }

  class PanoramaViewer {
    constructor(canvas) {
      this.canvas = canvas; this.yaw = 0; this.pitch = 0; this.fov = 78; this.loaded = false; this.version = 0;
      canvas.hidden = true;
      this.gl = canvas.getContext('webgl', {alpha: false, antialias: false});
      if (!this.gl) return;
      const gl = this.gl;
      const shader = (type, source) => { const result = gl.createShader(type); gl.shaderSource(result, source); gl.compileShader(result); if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error('미리보기 셰이더를 초기화하지 못했습니다.'); return result; };
      const vertex = shader(gl.VERTEX_SHADER, 'attribute vec2 point; varying vec2 screen; void main(){screen=point;gl_Position=vec4(point,0.,1.);}');
      const fragment = shader(gl.FRAGMENT_SHADER, `precision mediump float; varying vec2 screen; uniform float aspect, tangent, yaw, pitch; uniform sampler2D F,R,B,L,U,D;
        void main(){vec3 d=normalize(vec3(screen.x*aspect*tangent,screen.y*tangent,1.));d=vec3(d.x,cos(pitch)*d.y+sin(pitch)*d.z,-sin(pitch)*d.y+cos(pitch)*d.z);d=vec3(cos(yaw)*d.x+sin(yaw)*d.z,d.y,-sin(yaw)*d.x+cos(yaw)*d.z);vec3 a=abs(d);vec2 uv;
        if(a.z>=a.x&&a.z>=a.y){uv=vec2(d.z>0.?d.x:-d.x,d.y)/a.z*.5+.5;if(d.z>0.)gl_FragColor=texture2D(F,uv);else gl_FragColor=texture2D(B,uv);}
        else if(a.x>=a.y){uv=vec2(d.x>0.?-d.z:d.z,d.y)/a.x*.5+.5;if(d.x>0.)gl_FragColor=texture2D(R,uv);else gl_FragColor=texture2D(L,uv);}
        else{uv=vec2(d.x,d.y>0.?-d.z:d.z)/a.y*.5+.5;if(d.y>0.)gl_FragColor=texture2D(U,uv);else gl_FragColor=texture2D(D,uv);}}`);
      this.program = gl.createProgram(); gl.attachShader(this.program, vertex); gl.attachShader(this.program, fragment); gl.linkProgram(this.program);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error('미리보기 화면을 초기화하지 못했습니다.');
      gl.useProgram(this.program);
      const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]), gl.STATIC_DRAW);
      const point = gl.getAttribLocation(this.program, 'point'); gl.enableVertexAttribArray(point); gl.vertexAttribPointer(point, 2, gl.FLOAT, false, 0, 0);
      this.textures = Object.keys(faceNames).map((name, index) => { const texture = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + index); gl.bindTexture(gl.TEXTURE_2D, texture); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([22,31,28])); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); gl.uniform1i(gl.getUniformLocation(this.program, name), index); return texture; });
      this.locations = Object.fromEntries(['aspect','tangent','yaw','pitch'].map(key => [key, gl.getUniformLocation(this.program, key)]));
      let drag = null;
      canvas.addEventListener('pointerdown', event => { drag = {x: event.clientX, y: event.clientY}; canvas.setPointerCapture(event.pointerId); canvas.focus({preventScroll: true}); });
      canvas.addEventListener('pointermove', event => { if (!drag) return; this.yaw -= (event.clientX - drag.x) * .005; this.pitch = Math.max(-1.53, Math.min(1.53, this.pitch + (event.clientY - drag.y) * .005)); drag = {x:event.clientX,y:event.clientY}; this.draw(); });
      const release = () => { drag = null; }; canvas.addEventListener('pointerup', release); canvas.addEventListener('pointercancel', release); canvas.addEventListener('lostpointercapture', release);
      canvas.addEventListener('wheel', event => { if (!this.loaded) return; event.preventDefault(); this.zoom(event.deltaY > 0 ? 5 : -5); }, {passive:false});
      canvas.addEventListener('keydown', event => { const keys = ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-']; if (!keys.includes(event.key)) return; event.preventDefault(); if (event.key === 'ArrowLeft') this.yaw -= .1; if (event.key === 'ArrowRight') this.yaw += .1; if (event.key === 'ArrowUp') this.pitch = Math.min(1.53,this.pitch+.1); if (event.key === 'ArrowDown') this.pitch = Math.max(-1.53,this.pitch-.1); if (event.key === '+') this.zoom(-5); if (event.key === '-') this.zoom(5); this.draw(); });
      new ResizeObserver(() => this.draw()).observe(canvas);
    }
    async load(urls) {
      const version = ++this.version; this.loaded = false;
      if (!this.gl) { await loadFlatPreview(urls.F); this.loaded = true; return; }
      const images = await Promise.all(Object.keys(faceNames).map(face => new Promise((resolve,reject) => { const image = new Image(); const timer = setTimeout(() => { image.src = ''; reject(new Error('거리뷰 이미지 응답 시간이 초과되었습니다.')); }, 30000); image.onload = () => {clearTimeout(timer);resolve(image);}; image.onerror = () => {clearTimeout(timer);reject(new Error('거리뷰 타일을 불러오지 못했습니다. 지점을 다시 선택해 주세요.'));}; image.src = urls[face]; })));
      if (version !== this.version) return;
      const gl = this.gl; gl.useProgram(this.program); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      images.forEach((image,index) => {gl.activeTexture(gl.TEXTURE0 + index); gl.bindTexture(gl.TEXTURE_2D,this.textures[index]);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,image);});
      this.loaded = true; this.canvas.hidden = false; this.reset();
    }
    zoom(delta) { this.fov = Math.max(35,Math.min(110,this.fov+delta)); this.draw(); }
    reset() { this.yaw=0;this.pitch=0;this.fov=78;this.draw(); }
    draw() { if (!this.gl || !this.loaded) return; const gl=this.gl; const ratio=Math.min(window.devicePixelRatio||1,2); const width=Math.max(1,Math.round(this.canvas.clientWidth*ratio)),height=Math.max(1,Math.round(this.canvas.clientHeight*ratio)); if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;} gl.viewport(0,0,width,height);gl.useProgram(this.program);gl.uniform1f(this.locations.aspect,width/height);gl.uniform1f(this.locations.tangent,Math.tan(this.fov*Math.PI/360)/(width/height));gl.uniform1f(this.locations.yaw,this.yaw);gl.uniform1f(this.locations.pitch,this.pitch);gl.drawArrays(gl.TRIANGLES,0,6); }
  }
  function loadFlatPreview(url) {
    return new Promise((resolve, reject) => {
      const image = $('flat-preview'); image.hidden = true;
      const timer = setTimeout(() => { image.onload = null; image.onerror = null; reject(new Error('거리뷰 이미지 응답 시간이 초과되었습니다.')); }, 30000);
      image.onload = () => { clearTimeout(timer); image.onload = null; image.onerror = null; image.hidden = false; resolve(); };
      image.onerror = () => { clearTimeout(timer); image.onload = null; image.onerror = null; reject(new Error('거리뷰 이미지를 불러오지 못했습니다.')); };
      image.src = url;
    });
  }
  let viewer;
  try { viewer = new PanoramaViewer($('panorama-canvas')); } catch (_) { viewer = {gl:null,loaded:false,load:async urls=>loadFlatPreview(urls.F),zoom:()=>{},reset:()=>{}}; }
  async function openPanorama(pano) {
    const request = ++state.previewRequest; state.active = pano; renderStations(); renderMarkers(); updateControls();
    if (map) map.panTo([pano.lat,pano.lng], {animate:true});
    $('preview-loading').hidden=false; $('preview-placeholder').hidden=true; $('preview-controls').hidden=true; $('fallback-faces').hidden=true; $('panorama-canvas').hidden=true; $('flat-preview').hidden=true;
    text($('preview-title'), pano.title || pano.description || '촬영 지점'); text($('preview-meta'), captureLabel(pano));
    try {
      const details=await api(`/api/panorama?id=${encodeURIComponent(pano.id)}`,{timeout:45000}); if(request!==state.previewRequest)return;
      state.active={...pano,...details};
      updateNaverLink();
      const urls=Object.fromEntries(Object.keys(faceNames).map(face=>[face,details.faces?.[face]||`/api/cube/${encodeURIComponent(pano.id)}/${face}`]));
      await viewer.load(urls); if(request!==state.previewRequest)return;
      text($('preview-title'),details.title||details.description||'촬영 지점');
      text($('preview-meta'),`${captureLabel(details)} · ${Number(details.lat??pano.lat).toFixed(5)}, ${Number(details.lng??details.lon??pano.lng).toFixed(5)} · 저해상도 미리보기`);
      $('preview-controls').hidden=!viewer.gl; $('preview-fullscreen').disabled=false;
      if(!viewer.gl){const tabs=$('fallback-faces');tabs.replaceChildren();for(const[face,label]of Object.entries(faceNames)){const button=el('button','',label);button.addEventListener('click',async()=>{try{await loadFlatPreview(urls[face]);}catch(error){toast(error.message,true);}});tabs.append(button);}tabs.hidden=false;toast('이 브라우저에서는 방향별 사진 미리보기를 제공합니다.');}
    } catch(error){if(request!==state.previewRequest)return;toast(error.message,true);$('preview-placeholder').hidden=false;text($('preview-placeholder').querySelector('strong'),'거리뷰를 불러오지 못했습니다');text($('preview-placeholder').querySelector('p'),'지점을 다시 선택하여 시도해 주세요.');}
    finally{if(request===state.previewRequest){$('preview-loading').hidden=true;updateControls();}}
  }
  function payload() {
    const ids = inputIds();
    if(!state.snapshot||scopeDirty()||!validPolicy()||!ids.length)throw new Error('조회한 범위와 촬영 조건을 확인하고 사용할 지점을 선택해 주세요.');
    const result = {center:{...state.snapshot.center},radius_m:state.snapshot.radius_m,generation_mode:'multi_view',capture_policy:currentPolicy(),panorama_ids:ids,excluded_panorama_ids:state.filtered.filter(p => state.excluded.has(p.id)).map(p => p.id),processing_options:{remove_sky:$('remove-sky').checked,mask_dynamic:$('mask-dynamic').checked},size_filter:generationSizeOptions(),depth_cleanup:generationDepthOptions()};
    return window.streetviewWorkflow?.extendConfig?.(result) || result;
  }
  async function submit(kind) {
    if(state.submitting)return;
    try {const body=payload();state.submitting=true;updateControls();
      const response=await api(kind==='plan'?'/api/plans':'/api/jobs',{method:'POST',body:JSON.stringify(body),timeout:180000});
      if(kind==='plan'){const link=$('saved-link');link.href=response.download_url||`/api/plans/${encodeURIComponent(response.plan.id)}/download`;link.hidden=false;toast('선택한 설정을 저장했습니다. 다운로드할 수 있습니다.');}
      else{toast('생성 작업을 시작했습니다. 작업 기록에서 진행 상황을 확인하세요.');await refreshStatus();}
    }catch(error){toast(error.message,true);}finally{state.submitting=false;updateControls();}
  }
  const statusLabels={queued:'대기 중',running:'진행 중',cancelling:'취소 중',cancelled:'취소됨',completed:'파일 생성 완료',failed:'실패',interrupted:'중단됨'};
  const stageLabels={collect:'촬영 자료 수집',preprocess:'자료 전처리',sfm:'카메라·형상 복원',train:'Gaussian 학습',infer:'Gaussian 생성',export:'PLY 내보내기'};
  function renderJobs(jobs) {
    text($('job-count'),jobs.length);const list=$('job-list');
    if(!jobs.length){if(!list.querySelector('.jobs-empty')){list.replaceChildren();const empty=el('div','jobs-empty');empty.append(el('div','stack-symbol','▱\n▱\n▱'),el('strong','','아직 생성 작업이 없습니다'),el('p','','촬영 지점과 조건을 선택한 뒤\n설정을 저장하거나 생성을 시작하세요.'));list.append(empty);}return;}
    list.replaceChildren();
    for(const job of jobs){
      const jobMode = (job.config?.generation_mode || job.generation_mode) === 'single_panorama' ? '한 지점' : '주변 공간';
      const card=el('article','job-card');card.dataset.jobId=job.id;card.dataset.jobKind='generation';const title=el('div','job-title');title.append(el('strong','',`${jobMode} 생성 · ${String(job.id||'').slice(0,8)}`),el('span',`job-status ${job.status}`,statusLabels[job.status]||String(job.status||'상태 미확인')));card.append(title);
      const created=job.created_utc||job.created_at;const date=created?new Date(created):null;const stage=job.current_stage||job.stage;const stageText=stageLabels[typeof stage==='object'?stage?.name:stage]||'';
      card.append(el('p','job-meta',`${date&&!Number.isNaN(date.getTime())?date.toLocaleString('ko-KR',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'}):''}${stageText?' · '+stageText:''}`));
      const processing = job.config?.processing_options;
      if (typeof processing?.remove_sky === 'boolean' && typeof processing?.mask_dynamic === 'boolean') card.append(el('p','job-meta job-options',`${processing.remove_sky ? '하늘 제거' : '하늘 유지'} · ${processing.mask_dynamic ? '인물·차량 마스크 켬' : '인물·차량 마스크 끔'}`));
      const sizeFilter = job.config?.size_filter;
      if (typeof job.config?.depth_cleanup?.enabled === 'boolean' && jobMode !== '한 지점') card.append(el('p','job-meta job-options',`뎁스 기반 부유물 정리 ${job.config.depth_cleanup.enabled ? '켬' : '끔'}`));
      if (sizeFilter?.enabled === true && Number.isFinite(sizeFilter.max_sigma_camera_radius_ratio)) card.append(el('p','job-meta job-options',`큰 가우시안 제거 · 크기 제한 ${(sizeFilter.max_sigma_camera_radius_ratio * 100).toLocaleString()}%`));
      const raw=typeof job.progress==='number'?job.progress:(job.progress?.fraction??null);
      if(Number.isFinite(raw)){const fraction=Math.max(0,Math.min(1,raw>1?raw/100:raw));const progress=el('div','job-progress');progress.setAttribute('role','progressbar');progress.setAttribute('aria-valuenow',String(Math.round(fraction*100)));progress.setAttribute('aria-valuemin','0');progress.setAttribute('aria-valuemax','100');const fill=el('span');fill.style.width=`${fraction*100}%`;progress.append(fill);card.append(progress);}
      if(job.error||job.message)card.append(el('p','job-meta',String(job.error||job.message)));
      const actions=el('div','job-actions');
      if(job.status==='completed'&&job.artifact){const link=el('a','','PLY 다운로드 ↓');link.href=`/api/jobs/${encodeURIComponent(job.id)}/download`;actions.append(link);}
      if(['queued','running'].includes(job.status)){const cancel=el('button','','작업 취소');cancel.addEventListener('click',async()=>{cancel.disabled=true;try{await api(`/api/jobs/${encodeURIComponent(job.id)}/cancel`,{method:'POST',body:'{}'});await refreshStatus();}catch(error){toast(error.message,true);cancel.disabled=false;}});actions.append(cancel);}
      card.append(actions);list.append(card);
      window.streetviewWorkflow?.decorateJob?.(card,job,'generation');
    }
  }
  function updateFilterControls() {
    const active = filterState.jobs.some(job => ['queued', 'running'].includes(job.status));
    const pathsReady = $('ply-filter-source').value.trim() && $('ply-filter-cameras').value.trim();
    const ratioReady = (!$('ply-size-enabled').checked || validSizePercent($('ply-filter-percent').value)) && (!$('ply-crop-enabled').checked || validSizePercent($('ply-crop-percent').value));
    $('ply-filter-submit').disabled = !filterState.ready || filterState.submitting || active || !pathsReady || !ratioReady;
    text($('ply-filter-submit'), filterState.submitting ? '요청하는 중…' : active ? 'PLY 정리 중…' : '정리 PLY 만들기');
    for (const id of ['ply-filter-source', 'ply-filter-cameras', 'ply-filter-percent']) $(id).disabled = filterState.submitting;
    $('ply-filter-percent').setAttribute('aria-invalid', String(!ratioReady));
    $('ply-filter-percent').disabled = filterState.submitting || !$('ply-size-enabled').checked;
    $('ply-crop-percent').disabled = filterState.submitting || !$('ply-crop-enabled').checked;
  }
  function renderFilterJobs(jobs) {
    const list = $('ply-filter-jobs'); list.replaceChildren();
    if (!jobs.length) { list.append(el('p','helper','아직 정리한 PLY가 없습니다.')); return; }
    for (const job of jobs) {
      const card = el('article','job-card');
      card.dataset.jobId=job.id;card.dataset.jobKind='filter';
      const title = el('div','job-title');
      const sourceName = String(job.source_ply || '').split(/[\\/]/).pop();
      title.append(el('strong','',sourceName || `PLY 정리 · ${String(job.id || '').slice(0,8)}`), el('span',`job-status ${job.status}`,statusLabels[job.status] || '상태 확인 중'));
      card.append(title);
      const ratio = job.options?.max_sigma_camera_radius_ratio;
      if (job.options?.enabled !== false && Number.isFinite(ratio)) card.append(el('p','job-meta',`크기 제한 ${(ratio * 100).toLocaleString()}%`));
      if (job.crop?.enabled) card.append(el('p','job-meta',`출력 반경 ${(job.crop.radius_camera_radius_ratio*100).toLocaleString()}% · 높이 유지`));
      if (Number.isInteger(job.removed_rows) && Number.isInteger(job.remaining_rows)) card.append(el('p','job-meta',`${job.removed_rows.toLocaleString()}개 삭제 · ${job.remaining_rows.toLocaleString()}개 유지`));
      if (job.error) card.append(el('p','job-meta',String(job.error)));
      if (job.status === 'completed' && job.artifact) {
        const actions = el('div','job-actions'), link = el('a','','정리 PLY 다운로드 ↓');
        link.href = `/api/ply-filters/${encodeURIComponent(job.id)}/download`; actions.append(link); card.append(actions);
      }
      list.append(card);
      window.streetviewWorkflow?.decorateJob?.(card,job,'filter');
    }
  }
  async function refreshPlyFilters() {
    try {
      const response = await api('/api/ply-filters', {timeout: 12000});
      if (!Array.isArray(response.jobs)) throw new Error('정리 작업 목록을 확인할 수 없습니다.');
      filterState.ready = true; filterState.jobs = response.jobs;
      if (!filterState.defaultsApplied && !filterState.touched && !filterState.submitting && response.defaults) {
        if (typeof response.defaults.source_ply === 'string') $('ply-filter-source').value = response.defaults.source_ply;
        if (typeof response.defaults.camera_json === 'string') $('ply-filter-cameras').value = response.defaults.camera_json;
        if (typeof response.defaults.size_filter_enabled === 'boolean') $('ply-size-enabled').checked = response.defaults.size_filter_enabled;
        if (typeof response.defaults.crop?.enabled === 'boolean') $('ply-crop-enabled').checked = response.defaults.crop.enabled;
        if (Number.isFinite(response.defaults.crop?.radius_camera_radius_ratio)) $('ply-crop-percent').value = response.defaults.crop.radius_camera_radius_ratio * 100;
        const ratio = response.defaults.max_sigma_camera_radius_ratio;
        if (Number.isFinite(ratio) && validSizePercent(String(ratio * 100))) $('ply-filter-percent').value = ratio * 100;
        filterState.defaultsApplied = true;
      }
      renderFilterJobs(filterState.jobs);
      text($('ply-filter-note'), filterState.jobs.some(job => ['queued','running'].includes(job.status)) ? '파일을 정리하고 있습니다. 작업이 끝나면 다운로드할 수 있습니다.' : '이 컴퓨터의 파일을 사용합니다. 버튼을 누르면 정리를 시작합니다.');
    } catch (_) {
      filterState.ready = false;
      text($('ply-filter-note'),'PLY 정리 기능에 연결할 수 없습니다. 서버 상태를 확인해 주세요.');
    } finally { updateFilterControls(); }
  }
  async function submitPlyFilter(event) {
    event.preventDefault();
    if ($('ply-filter-submit').disabled || filterState.submitting) return;
    const source_ply = $('ply-filter-source').value.trim(), camera_json = $('ply-filter-cameras').value.trim();
    const value = $('ply-filter-percent').value;
    if (($('ply-size-enabled').checked && !validSizePercent(value)) || ($('ply-crop-enabled').checked && !validSizePercent($('ply-crop-percent').value))) { toast('크기 제한과 출력 반경은 0.01~1,000%로 입력해 주세요.',true); return; }
    try {
      filterState.submitting = true; updateFilterControls();
      const body={source_ply,camera_json,max_sigma_camera_radius_ratio:validSizePercent(value)?Number(value)/100:.5};
      if (!$('ply-size-enabled').checked) body.size_filter_enabled=false;
      if ($('ply-crop-enabled').checked) body.crop={enabled:true,radius_camera_radius_ratio:Number($('ply-crop-percent').value)/100};
      await api('/api/ply-filters', {method:'POST', body:JSON.stringify(body), timeout:45000});
      toast('PLY 정리를 시작했습니다. 원본은 보관합니다.');
      await refreshPlyFilters();
    } catch(error) { toast(error.message,true); }
    finally { filterState.submitting = false; updateFilterControls(); }
  }
  async function refreshStatus() {
    if(statusBusy)return;statusBusy=true;
    try{state.status=await api('/api/status',{timeout:12000});$('connection-dot').className='status-dot online';text($('connection-label'),'로컬 서버 연결됨');renderJobs(Array.isArray(state.status.jobs)?state.status.jobs:[]);}
    catch(error){state.status=null;$('connection-dot').className='status-dot offline';text($('connection-label'),'서버 연결 확인 필요');}
    finally{await refreshPlyFilters();statusBusy=false;updateControls();}
  }

  const rescan=el('button','text-button','선택한 날짜로 다시 조회 ↻');rescan.id='rescan-date';rescan.type='button';rescan.disabled=true;rescan.style.cssText='display:block;margin-top:9px;color:var(--mint)';$('capture-date').after(rescan);
  $('coordinate-form').addEventListener('submit',event=>{event.preventDefault();if(!$('latitude').value||!$('longitude').value){toast('위도와 경도를 모두 입력해 주세요.',true);return;}setCenter($('latitude').value,$('longitude').value,true);});
  for(const id of ['latitude','longitude'])$(id).addEventListener('input',()=>{state.coordinateDraft=true;updateControls();});
  $('radius').addEventListener('input',event=>{state.radius=Number(event.target.value);$('radius-value').value=state.radius;drawScope();renderStations();updateControls();});
  $('radius-value').addEventListener('input',event=>{state.radius=event.target.value===''?NaN:Number(event.target.value);syncRadiusSlider();drawScope();renderStations();updateControls();});
  $('radius-value').addEventListener('change',event=>{if(!Number.isFinite(state.radius))return;state.radius=Math.max(10,state.radius);event.target.value=state.radius;syncRadiusSlider();drawScope();renderStations();updateControls();});
  $('discover').addEventListener('click',()=>discover());rescan.addEventListener('click',()=>discover(true));$('fit-area').addEventListener('click',()=>drawScope(true));
  $('load-more').addEventListener('click',()=>discover(false, true));
  document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{const previous=$('capture-date').value;state.mode=button.dataset.mode;syncModeButtons();populateDates(state.mode==='same_month'?previous.slice(0,7):previous);applyFilters(true);}));
  $('capture-date').addEventListener('change',()=>applyFilters(true));
  for (const id of ['remove-sky', 'mask-dynamic', 'depth-cleanup-enabled']) $(id).addEventListener('change',()=>{$('saved-link').hidden=true;updateControls();});
  for (const id of ['size-filter-enabled', 'size-filter-percent']) $(id).addEventListener('input',()=>{$('saved-link').hidden=true;updateControls();});
  $('ply-filter-form').addEventListener('submit',submitPlyFilter);
  for (const id of ['ply-filter-source', 'ply-filter-cameras', 'ply-filter-percent','ply-size-enabled','ply-crop-enabled','ply-crop-percent']) $(id).addEventListener('input',()=>{filterState.touched=true;updateFilterControls();});
  $('select-all').addEventListener('click',()=>{for(const pano of state.filtered){state.selected.add(pano.id);state.manualIncluded.add(pano.id);state.excluded.delete(pano.id);}$('saved-link').hidden=true;renderStations();renderMarkers();updateControls();});
  $('clear-selection').addEventListener('click',()=>{for(const pano of state.filtered){state.excluded.add(pano.id);state.manualIncluded.delete(pano.id);}state.selected.clear();$('saved-link').hidden=true;renderStations();renderMarkers();updateControls();});
  $('toggle-preview-selection').addEventListener('click',()=>{if(state.active)toggleSelection(state.active.id);});
  $('save-plan').addEventListener('click',()=>submit('plan'));$('generate').addEventListener('click',()=>{if(canGenerate())submit('job');});
  $('refresh-jobs').addEventListener('click',refreshStatus);$('zoom-in').addEventListener('click',()=>viewer.zoom(-8));$('zoom-out').addEventListener('click',()=>viewer.zoom(8));$('reset-view').addEventListener('click',()=>viewer.reset());
  $('preview-fullscreen').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else if($('preview-stage').requestFullscreen)await $('preview-stage').requestFullscreen();}catch(_){toast('전체 화면을 열지 못했습니다.',true);}});
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)refreshStatus();});
  window.streetviewStudio={getConfig:payload,refresh:refreshStatus,refreshControls:updateControls,toast,
    applySettings(settings){
      if(settings.generation_mode && settings.generation_mode !== 'multi_view')throw new Error('주변 공간 프리셋만 사용할 수 있습니다.');
      if(settings.processing_options){$('remove-sky').checked=!!settings.processing_options.remove_sky;$('mask-dynamic').checked=!!settings.processing_options.mask_dynamic;}
      if(settings.size_filter){$('size-filter-enabled').checked=!!settings.size_filter.enabled;$('size-filter-percent').value=settings.size_filter.max_sigma_camera_radius_ratio*100;}
      if(settings.depth_cleanup){$('depth-cleanup-enabled').checked=!!settings.depth_cleanup.enabled;depthState.defaultApplied=true;}
      updateControls();
    },
    setCleanupInputs(value){$('ply-filter-source').value=value.source_ply;$('ply-filter-cameras').value=value.camera_json;filterState.touched=true;$('ply-size-enabled').checked=false;updateFilterControls();$('ply-filter-form').scrollIntoView({behavior:'smooth',block:'center'});},
    jobs(){return{generation:state.status?.jobs||[],filter:filterState.jobs};}};
  setupMap();updateControls();updateFilterControls();refreshStatus();setInterval(()=>{if(!document.hidden)refreshStatus();},5000);
})();
