(() => {
  'use strict';

  const STAGES = ['mine', 'craft', 'patch'];
  const STAGE_NAMES = { mine: 'Explore', craft: 'Build a task', patch: 'See a repair' };
  const BEHAVIOR_LABELS = {
    manual_login: 'Log in',
    manual_signup: 'Sign up',
    create_delivery_operations_project: 'Create a project',
    create_board_and_rename_team: 'Create board + rename team',
    create_list_and_add_card: 'Add list + card',
    create_team_board: 'Create a team board',
    demote_member_and_create_task_board: 'Change role + create task board',
    create_escalation_card_with_details: 'Create a detailed card',
    create_resolved_list_with_cards: 'Create list + move card',
    open_comment_for_editing: 'Open comment editor',
    create_workflow_board_and_organize_cards: 'Set up a board and cards',
    view_card_and_column_editors: 'Open card and column editors',
    remove_val_acc_bar_panel: 'Remove validation accuracy panel'
  };
  const CHECK_NAMES = {
    pass: 'Passed',
    'expected-failure': 'Expected failure',
    unknown: 'Unknown'
  };
  const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
  const isText = (value) => typeof value === 'string';
  const isLabel = (value) => isText(value) && value.trim().length > 0;
  const isCount = (value) => Number.isSafeInteger(value) && value >= 0;
  const isCodeFiles = (files) => Array.isArray(files) && files.length > 0 && files.every((file) =>
    isObject(file) && isLabel(file.path) && isText(file.diff) && isCount(file.removedLines));

  function requireData(condition, message) {
    if (!condition) throw new Error(`Invalid explorer data: ${message}.`);
  }

  function assetURL(path, manifestURL, publicMediaBase = null) {
    requireData(
      isLabel(path) && !path.includes('\\'),
      'media paths must be valid site URLs'
    );
    const url = new URL(path, manifestURL);
    const base = publicMediaBase ? new URL(publicMediaBase) : null;
    const published = base && base.protocol === 'https:' && !base.username && !base.password &&
      url.origin === base.origin &&
      url.pathname.startsWith(base.pathname.endsWith('/') ? base.pathname : `${base.pathname}/`);
    requireData(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password &&
      (url.origin === manifestURL.origin || published), 'media must use this site or the configured publication base');
    return url.href;
  }

  function validateManifest(data, manifestURL) {
    requireData(isObject(data) && data.version === 1, 'a version 1 manifest is required');
    requireData(typeof data.preview === 'boolean', 'preview must be a boolean');
    requireData(Array.isArray(data.cases) && data.cases.length > 0, 'cases must be a nonempty array');
    const ids = new Set();
    data.cases.forEach((item, index) => {
      const label = `case ${index + 1}`;
      requireData(isObject(item), `${label} must be an object`);
      requireData(['id', 'title', 'app'].every((key) => isLabel(item[key])), `${label} needs an id, title, and app`);
      requireData(!ids.has(item.id), `${label} has a duplicate id`);
      ids.add(item.id);
      requireData(isCount(item.depth) && item.depth > 0, `${label} depth must be a positive integer`);
      requireData(['Logic only', 'Logic + UI'].includes(item.scope), `${label} scope is not supported`);
      requireData(isText(item.brief), `${label} brief must be text`);
      requireData(
        isObject(item.source) && isText(item.source.revision) && isText(item.source.instanceId),
        `${label} needs source revision and instanceId`
      );
      requireData(
        Array.isArray(item.lineage) && item.lineage.every((node) =>
          isObject(node) && isLabel(node.name) && isLabel(node.label) && typeof node.masked === 'boolean'),
        `${label} lineage must contain named setup or repair targets`
      );
      requireData(
        isObject(item.mine) && isText(item.mine.description) &&
        isCount(item.mine.actions) && typeof item.mine.verified === 'boolean',
        `${label} Mine fields are incomplete`
      );
      requireData(
        isObject(item.craft) && isText(item.craft.description) && isText(item.craft.diff) &&
        isCount(item.craft.filesChanged) && isCount(item.craft.removedLines) &&
        Array.isArray(item.craft.checks),
        `${label} Craft fields are incomplete`
      );
      requireData(item.craft.checks.every((check) =>
        isObject(check) && isLabel(check.label) &&
        isText(check.status) &&
        Object.prototype.hasOwnProperty.call(CHECK_NAMES, check.status) &&
        (check.detail === undefined || isText(check.detail))),
      `${label} has an invalid verification check`);
      requireData(
        isObject(item.patch) && isLabel(item.patch.model) &&
        typeof item.patch.binarySuccess === 'boolean' &&
        Number.isFinite(item.patch.chainScore) && isCount(item.patch.frameCount),
        `${label} Patch fields are incomplete`
      );
      requireData(isObject(item.graph) && Array.isArray(item.graph.nodes) && item.graph.nodes.length > 0,
        `${label} needs a mined behavior graph`);
      const nodes = new Map();
      item.graph.nodes.forEach((node) => {
        requireData(isObject(node) && isLabel(node.id) && isLabel(node.label) &&
          !nodes.has(node.id) && (node.parent === null || isLabel(node.parent)) &&
          isCount(node.depth) && node.depth > 0 && isCount(node.actions) &&
          isText(node.description) && typeof node.onPath === 'boolean' && typeof node.masked === 'boolean' &&
          (node.round === null || node.round === 'parallel' || isCount(node.round)),
        `${label} has an invalid graph node`);
        nodes.set(node.id, node);
      });
      nodes.forEach((node) => {
        requireData(node.parent === null ||
          (nodes.has(node.parent) && nodes.get(node.parent).depth < node.depth),
        `${label} has a missing or cyclic prerequisite`);
        const stage = item.lineage.find((entry) => entry.name === node.id);
        requireData(node.onPath === Boolean(stage) && node.masked === Boolean(stage && stage.masked),
          `${label} graph roles disagree with the recorded lineage`);
      });
      requireData(item.lineage.length > 0 && new Set(item.lineage.map((node) => node.name)).size === item.lineage.length &&
        item.lineage.every((node, position) => nodes.has(node.name) &&
          nodes.get(node.name).parent === (position === 0 ? null : item.lineage[position - 1].name)) &&
        item.graph.target === item.lineage[item.lineage.length - 1].name,
      `${label} lineage must follow the prerequisite edges to the selected target`);
      const maskedNames = item.lineage.filter((node) => node.masked).map((node) => node.name);
      requireData(maskedNames.length === item.depth && isObject(item.composition) &&
        ['atomic', 'cumulative'].includes(item.composition.kind) &&
        ['Single validated mask', 'Agent-assisted merge', 'Deterministic merge'].includes(item.composition.method) &&
        isCount(item.composition.conflicts) && Array.isArray(item.composition.components) &&
        item.composition.components.length === item.depth &&
        item.composition.components.every((component, position) => isObject(component) &&
          isLabel(component.id) && component.trace === maskedNames[position] &&
          component.verified === true && isCount(component.files)) &&
        new Set(item.composition.components.map((component) => component.id)).size === item.depth,
      `${label} needs the exact verified component masks`);
      requireData(item.composition.components.every((component) =>
        isCodeFiles(component.diffFiles) && isText(component.brief) && isCount(component.removedLines)) &&
        Array.isArray(item.craft.examples) && item.craft.examples.length > 0 &&
        item.craft.examples.every((example) => isObject(example) && isLabel(example.id) &&
          isCount(example.depth) && example.depth > 0 && example.depth <= item.depth &&
          Array.isArray(example.traces) && example.traces.length === example.depth &&
          example.traces.every((name, position) => name === maskedNames[position]) &&
          isText(example.brief) && isCodeFiles(example.diffFiles) &&
          example.diffFiles.some((file) => file.path === example.targetFile) &&
          ['Single validated mask', 'Agent-assisted merge', 'Deterministic merge'].includes(example.method) &&
          isCount(example.filesChanged) && isCount(example.removedLines) && isCount(example.conflicts)),
      `${label} needs recorded code masks and matching generated task examples`);
      const mediaItems = [item.mine.media, item.patch.media];
      nodes.forEach((node) => {
        requireData(node.media === null || isObject(node.media), `${label} must explicitly identify available node media`);
        if (node.media !== null) mediaItems.push(node.media);
      });
      mediaItems.forEach((media) => {
        requireData(
          isObject(media) && ['gif', 'video'].includes(media.kind) && isCount(media.bytes),
          `${label} has invalid replay media`
        );
        assetURL(media.src, manifestURL);
        assetURL(media.poster, manifestURL);
      });
    });
    return data;
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(className, text) {
    const node = element('button', `pd-button ${className}`, text);
    node.type = 'button';
    return node;
  }

  function behaviorLabel(node) {
    const id = node.id || node.name;
    return Object.prototype.hasOwnProperty.call(BEHAVIOR_LABELS, id) ? BEHAVIOR_LABELS[id] : node.label;
  }

  function appLabel(item) {
    return item.application?.label || item.app;
  }

  async function loadApplicationSources(path, signal) {
    requireData(isLabel(path), 'an application source catalog is required');
    const url = new URL(path, document.baseURI);
    requireData(url.origin === location.origin, 'application sources must be served by this site');
    const response = await fetch(url, { credentials: 'same-origin', mode: 'same-origin', cache: 'no-cache', signal });
    if (!response.ok) throw new Error(`Application sources returned HTTP ${response.status}.`);
    const data = await response.json();
    requireData(data.version === 1 && Array.isArray(data.applications), 'invalid application source catalog');
    const sources = new Map();
    data.applications.forEach((app) => {
      requireData(isObject(app) && isLabel(app.id) && isLabel(app.label) &&
        typeof app.repository === 'string' && /^[A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+$/.test(app.repository) &&
        !['.', '..'].includes(app.repository.split('/')[1]) &&
        typeof app.commit === 'string' && /^[a-f0-9]{7,40}$/.test(app.commit) &&
        ['clone', 'benchmark', 'open_source'].includes(app.kind), 'invalid application provenance');
      requireData(!sources.has(app.id), 'duplicate application provenance');
      sources.set(app.id, { ...app, url: `https://github.com/${app.repository}` });
    });
    return sources;
  }

  function createApplicationSource(source) {
    const row = element('p', 'pd-app-source');
    const kinds = { clone: 'Open-source clone', benchmark: 'Benchmark app', open_source: 'Open-source application' };
    row.append(document.createTextNode(`${kinds[source.kind]} · GitHub `));
    const link = element('a', '', source.repository);
    link.href = source.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.setAttribute('aria-label', `${source.repository} source on GitHub (opens in a new tab)`);
    row.append(link);
    return row;
  }

  function createDialog(root, prefix) {
    const dialog = element('dialog', 'pd-dialog');
    const header = element('div', 'pd-dialog-header');
    const heading = element('div');
    const category = element('p', 'pd-label');
    const title = element('h3', 'pd-dialog-title');
    title.id = `${prefix}-dialog-title`;
    dialog.setAttribute('aria-labelledby', title.id);
    const closeButton = button('pd-button-secondary pd-dialog-close', 'Close');
    heading.append(category, title);
    header.append(heading, closeButton);
    const body = element('div', 'pd-dialog-body');
    dialog.append(header, body);
    root.append(dialog);
    let cleanup = null;
    let trigger = null;
    let previousOverflow = '';
    let previousScroll = { left: 0, top: 0 };
    let closeTouch = null;

    function close(restoreFocus = true) {
      if (!dialog.open) return;
      closeTouch = null;
      dialog.close();
      if (cleanup) cleanup();
      cleanup = null;
      body.replaceChildren();
      document.body.style.overflow = previousOverflow;
      if (restoreFocus && trigger && trigger.isConnected) trigger.focus({ preventScroll: true });
      window.scrollTo({ ...previousScroll, behavior: 'instant' });
    }

    // A post-pan tap can lose its compatibility click. Handle touch releases directly.
    closeButton.addEventListener('pointerdown', (event) => {
      closeTouch = event.pointerType === 'touch' && event.isPrimary
        ? { id: event.pointerId, x: event.clientX, y: event.clientY } : null;
      if (closeTouch) event.preventDefault();
    });
    closeButton.addEventListener('pointermove', (event) => {
      if (closeTouch && event.pointerId === closeTouch.id &&
        Math.hypot(event.clientX - closeTouch.x, event.clientY - closeTouch.y) > 10) closeTouch = null;
    });
    closeButton.addEventListener('pointercancel', () => { closeTouch = null; });
    closeButton.addEventListener('lostpointercapture', () => { closeTouch = null; });
    closeButton.addEventListener('pointerup', (event) => {
      const touch = closeTouch;
      closeTouch = null;
      if (!touch || event.pointerId !== touch.id) return;
      const bounds = closeButton.getBoundingClientRect();
      if (event.clientX >= bounds.left && event.clientX <= bounds.right &&
        event.clientY >= bounds.top && event.clientY <= bounds.bottom &&
        Math.hypot(event.clientX - touch.x, event.clientY - touch.y) <= 10) {
        event.preventDefault();
        close();
      }
    });
    closeButton.addEventListener('click', (event) => {
      if (event.pointerType === 'touch') event.preventDefault();
      else close();
    });
    dialog.addEventListener('cancel', (event) => {
      event.preventDefault();
      close();
    });
    dialog.addEventListener('keydown', (event) => {
      if (event.key !== 'Tab') return;
      const controls = Array.from(dialog.querySelectorAll(
        'a[href], button, input, select, textarea, summary, video[controls], [tabindex]'
      )).filter((node) => !node.disabled && node.tabIndex >= 0 && node.getClientRects().length > 0);
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
    dialog.addEventListener('click', (event) => {
      const bounds = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < bounds.left || event.clientX > bounds.right ||
        event.clientY < bounds.top || event.clientY > bounds.bottom)) close();
    });
    return {
      close,
      open(options) {
        const originalTrigger = dialog.open ? trigger : document.activeElement;
        close(false);
        category.textContent = options.category;
        category.hidden = !options.category;
        title.textContent = options.title;
        heading.classList.toggle('pd-sr-only', Boolean(options.hideHeading));
        body.replaceChildren(options.body);
        cleanup = options.cleanup || null;
        trigger = options.trigger && options.trigger.isConnected ? options.trigger : originalTrigger;
        previousOverflow = document.body.style.overflow;
        previousScroll = { left: window.scrollX, top: window.scrollY };
        document.body.style.overflow = 'hidden';
        dialog.showModal();
        body.scrollTop = 0;
        closeButton.focus({ preventScroll: true });
        window.scrollTo({ ...previousScroll, behavior: 'instant' });
      }
    };
  }

  function initializeFigureZoom(root, prefix) {
    const article = root.closest('article.blog-content');
    if (!article || article.dataset.pdFigureZoom) return;
    article.dataset.pdFigureZoom = 'true';
    const dialog = createDialog(root, `${prefix}-figure`);

    function isImageLink(link) {
      return link && article.contains(link) && link.closest('.post-figure') &&
        link.querySelector('img') && !link.hasAttribute('download') &&
        ['http:', 'https:'].includes(link.protocol) && link.origin === location.origin &&
        !link.username && !link.password && /\.(png|jpe?g|gif|webp|avif|svg)$/i.test(link.pathname);
    }

    function labelLinks() {
      article.querySelectorAll('.post-figure a[href]').forEach((link) => {
        if (!isImageLink(link)) return;
        link.setAttribute('aria-haspopup', 'dialog');
        link.setAttribute('aria-label', `Zoom image: ${link.querySelector('img').alt || 'Article figure'}`);
      });
    }
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', labelLinks, { once: true });
    } else {
      labelLinks();
    }

    // Delegate so figures after this async script (including multi-panel figures) work too.
    article.addEventListener('click', (event) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey ||
        event.shiftKey || event.altKey) return;
      const link = event.target.closest('a[href]');
      if (!isImageLink(link)) return;
      const source = assetURL(link.href, new URL(document.baseURI));
      event.preventDefault();
      const original = link.querySelector('img');
      const figure = link.closest('.post-figure');
      const caption = figure.querySelector('figcaption');
      const body = element('div', 'pd-figure-zoom');
      const retry = button('pd-button-secondary', 'Retry image');
      retry.hidden = true;
      const feedback = element('p', 'pd-note', 'Loading image...');
      feedback.setAttribute('role', 'status');
      const viewport = element('div', 'pd-figure-viewport');
      viewport.tabIndex = 0;
      viewport.setAttribute('role', 'region');
      viewport.setAttribute('aria-label', 'Enlarged figure. At actual size, scroll or swipe to pan.');
      const image = element('img', 'pd-figure-image');
      image.alt = original.alt;
      image.draggable = false;
      image.setAttribute('role', 'button');
      image.setAttribute('aria-label', `Enlarge figure: ${original.alt}`);
      image.setAttribute('aria-pressed', 'false');
      image.onload = () => {
        feedback.hidden = true;
        retry.hidden = true;
        image.setAttribute('aria-disabled', 'false');
        image.tabIndex = 0;
      };
      image.onerror = () => {
        feedback.textContent = 'Could not load the enlarged image. Retry or close to return to the article.';
        feedback.hidden = false;
        retry.hidden = false;
        image.setAttribute('aria-disabled', 'true');
        image.tabIndex = -1;
      };
      function loadImage() {
        const retryHadFocus = document.activeElement === retry;
        feedback.textContent = 'Loading image...';
        feedback.hidden = false;
        retry.hidden = true;
        image.setAttribute('aria-disabled', 'true');
        image.tabIndex = -1;
        image.removeAttribute('src');
        image.src = source;
        if (retryHadFocus) viewport.focus({ preventScroll: true });
      }
      retry.addEventListener('click', loadImage);
      function toggleSize() {
        if (image.getAttribute('aria-disabled') === 'true') return;
        const actualSize = viewport.classList.toggle('pd-figure-actual-size');
        image.setAttribute('aria-pressed', String(actualSize));
        image.setAttribute('aria-label', `${actualSize ? 'Fit' : 'Enlarge'} figure: ${original.alt}`);
        viewport.scrollTop = 0;
        viewport.scrollLeft = 0;
      }
      image.addEventListener('click', toggleSize);
      image.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        event.preventDefault();
        toggleSize();
      });
      viewport.append(image);
      body.append(retry, feedback, viewport);
      if (caption) body.append(element('p', 'pd-figure-caption', caption.textContent.trim()));
      dialog.open({
        title: 'Figure zoom', hideHeading: true, category: '', body, trigger: link,
        cleanup: () => {
          image.onload = null;
          image.onerror = null;
          image.removeAttribute('src');
        }
      });
      loadImage();
    });
  }

  function validateTraceExamples(data, url, sources) {
    requireData(isObject(data) && data.version === 1 &&
      Array.isArray(data.examples) && data.examples.length > 0, 'a trace example catalog is required');
    const ids = new Set();
    data.examples.forEach((example) => {
      requireData(isObject(example) &&
        ['id', 'appId', 'label', 'description', 'name', 'prerequisite'].every((key) => isLabel(example[key])) &&
        (example.parent_trace === null || isLabel(example.parent_trace)) &&
        !ids.has(example.id) && sources.has(example.appId) && example.status === 'validated_pass',
      'trace examples must be verified and identify their source application');
      ids.add(example.id);
      requireData(Array.isArray(example.action_trace) && example.action_trace.length > 0,
        'recorded trace actions are required');
      example.action_trace.forEach((action) => {
        requireData(isObject(action) && ['click', 'focus', 'press', 'type'].includes(action.action) &&
          (action.action === 'press' ? isLabel(action.key) : isLabel(action.selector)) &&
          (action.action !== 'type' || isText(action.text)),
        'unsupported recorded browser action');
      });
      const signals = example.expected_signals;
      requireData(isObject(signals) &&
        Object.keys(signals).every((key) =>
          ['all_visible_text', 'all_visible_text_sources', 'element_states'].includes(key)) &&
        Array.isArray(signals.all_visible_text) && signals.all_visible_text.every(isLabel) &&
        Array.isArray(signals.all_visible_text_sources) &&
        signals.all_visible_text_sources.length === signals.all_visible_text.length &&
        signals.all_visible_text_sources.every((source) => ['rendered_text', 'form_value'].includes(source)) &&
        Array.isArray(signals.element_states) &&
        signals.element_states.every((state) => isObject(state) && state.version === 1 &&
          ['selector', 'attribute', 'value'].every((key) => isLabel(state[key]))) &&
        signals.all_visible_text.length + signals.element_states.length > 0,
      'trace examples need supported recorded success signals');
      requireData(isObject(example.source) &&
        example.source.schema === 'minepatch.golden_behavior_trace.v2' &&
        isText(example.source.sha256) && /^[a-f0-9]{64}$/.test(example.source.sha256),
      'trace provenance is incomplete');
      requireData(isObject(example.media) && example.media.kind === 'video' &&
        isCount(example.media.bytes) && example.media.bytes > 0, 'a recorded reference video is required');
      assetURL(example.media.src, url);
      assetURL(example.media.poster, url);
    });
    return data;
  }

  function initializeTraceExamples(root, prefix) {
    const triggers = Array.from(document.querySelectorAll(`[data-pd-trace-examples="${root.id}"]`));
    if (!triggers.length) return;
    const dialog = createDialog(root, `${prefix}-trace`);
    let cached = null;

    triggers.forEach((trigger) => trigger.addEventListener('click', () => {
      const body = element('div', 'pd-trace-examples');
      const controller = new AbortController();
      let player = null;
      let loading = false;
      dialog.open({
        title: 'Recorded trace examples', category: '', body, trigger,
        cleanup: () => {
          controller.abort();
          if (player) player.destroy();
        }
      });

      function showExamples(data, url, sources) {
        const label = element('label', 'pd-label', 'Choose an example');
        const picker = element('select', 'pd-select');
        picker.id = `${prefix}-trace-example`;
        label.htmlFor = picker.id;
        data.examples.forEach((example) => {
          const option = element('option', '', example.label);
          option.value = example.id;
          picker.append(option);
        });
        const detail = element('div', 'pd-trace-detail');
        body.replaceChildren(label, picker, detail);

        function showExample() {
          if (player) player.destroy();
          const example = data.examples[picker.selectedIndex];
          detail.dataset.traceName = example.name;
          detail.replaceChildren(
            createApplicationSource(sources.get(example.appId)),
            element('p', 'pd-dialog-lead', example.description),
            element('h4', 'pd-dialog-section-title', 'First prepare the state'),
            element('p', 'pd-description', example.prerequisite),
            element('h4', 'pd-dialog-section-title', 'Recorded browser actions')
          );
          const actions = element('ol', 'pd-trace-actions');
          example.action_trace.forEach((action) => {
            const item = element('li');
            const target = action.action === 'press' ? action.key : action.selector;
            const input = action.action === 'type' ? ` ${JSON.stringify(action.text)}` : '';
            item.append(element('code', '', `${action.action} ${target}${input}`));
            actions.append(item);
          });
          detail.append(actions, element('h4', 'pd-dialog-section-title', 'What replay checks'));
          const checks = element('ul', 'pd-trace-checks');
          const checkGroups = new Map();
          function addCheck(label, content) {
            if (!checkGroups.has(label)) {
              const group = element('li');
              const values = element('div', 'pd-trace-values');
              group.append(element('span', 'pd-label', label), values);
              checks.append(group);
              checkGroups.set(label, values);
            }
            checkGroups.get(label).append(content);
          }
          example.expected_signals.all_visible_text.forEach((value, index) => {
            addCheck(example.expected_signals.all_visible_text_sources[index] === 'form_value'
              ? 'Expected form values' : 'Expected visible text', element('code', '', value));
          });
          example.expected_signals.element_states.forEach((state) => {
            const item = element('div');
            item.append(element('code', '', state.selector),
              element('p', 'pd-description', `${state.attribute} must equal ${state.value}`));
            addCheck('Expected element states', item);
          });
          detail.append(checks, element('p', 'pd-note',
            'These signals are checked after the recorded actions. Completing the clicks alone is not a pass.'));
          const recording = element('details', 'pd-trace-recording');
          recording.append(element('summary', '', 'Watch the reference recording'));
          const replay = createPlayer(example.media, url, 'Reference recording', example.label);
          player = replay;
          recording.append(replay.element);
          recording.addEventListener('toggle', () => {
            if (!recording.open) replay.stop();
          });
          const fields = element('details', 'pd-trace-fields');
          fields.append(element('summary', '', 'View the recorded fields'),
            element('pre', '', JSON.stringify({
              name: example.name,
              parent_trace: example.parent_trace,
              action_trace: example.action_trace,
              expected_signals: example.expected_signals
            }, null, 2)));
          detail.append(recording, fields);
        }
        picker.addEventListener('change', showExample);
        showExample();
      }

      async function load() {
        if (loading) return;
        loading = true;
        const message = element('p', 'pd-description', 'Loading recorded traces...');
        message.setAttribute('role', 'status');
        body.replaceChildren(message);
        try {
          requireData(isLabel(trigger.dataset.tracesSrc), 'a trace example URL is required');
          const url = new URL(trigger.dataset.tracesSrc, document.baseURI);
          requireData(url.origin === location.origin && ['http:', 'https:'].includes(url.protocol) &&
            !url.username && !url.password, 'trace examples must be served by this site');
          if (!cached || cached.url.href !== url.href) {
            const response = await fetch(url, {
              credentials: 'same-origin', mode: 'same-origin', cache: 'no-cache',
              signal: controller.signal
            });
            if (!response.ok) throw new Error(`Trace examples returned HTTP ${response.status}.`);
            requireData(new URL(response.url || url.href).origin === url.origin,
              'trace examples redirected to another site');
            const data = await response.json();
            const sources = await loadApplicationSources(root.dataset.applicationSourcesSrc, controller.signal);
            validateTraceExamples(data, url, sources);
            if (controller.signal.aborted) return;
            cached = { data, url, sources };
          }
          if (!controller.signal.aborted) showExamples(cached.data, cached.url, cached.sources);
        } catch (error) {
          if (controller.signal.aborted) return;
          const reason = error instanceof Error ? error.message : 'An unexpected loading error occurred.';
          const message = element('p', 'pd-description', `The trace examples could not load. ${reason}`);
          message.setAttribute('role', 'alert');
          const retry = button('pd-button-secondary', 'Retry');
          retry.addEventListener('click', load);
          body.replaceChildren(message, retry);
        } finally {
          loading = false;
        }
      }
      load();
    }));
  }

  function status(text, kind) {
    return element('span', `pd-status pd-status-${kind}`, text);
  }

  function stats(entries) {
    const list = element('dl', 'pd-stats');
    entries.forEach(([label, value, modifier]) => {
      const item = element('div', `pd-stat${modifier ? ` pd-stat-${modifier}` : ''}`);
      item.append(element('dt', '', label), element('dd', '', String(value)));
      list.append(item);
    });
    return list;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function createPlayer(media, manifestURL, title, caseTitle, publicMediaBase = null, options = {}) {
    const replayURL = assetURL(media.src, manifestURL, publicMediaBase);
    const container = element('div', 'pd-player');
    const nativeControls = media.kind === 'video' && options.controls === true;

    const frame = element('div', 'pd-media-frame');
    if (Number.isFinite(media.width) && media.width > 0 && Number.isFinite(media.height) && media.height > 0) {
      frame.style.aspectRatio = `${media.width} / ${media.height}`;
    }
    const placeholder = element('span', 'pd-media-placeholder', 'Loading preview...');
    const poster = element('img', 'pd-media');
    poster.alt = `${title}: still preview of ${caseTitle}`;
    poster.loading = 'lazy';
    poster.decoding = 'async';
    const play = button('pd-play', '');
    const icon = element('span', 'pd-play-icon');
    icon.setAttribute('aria-hidden', 'true');
    const playLabel = element('span', '', 'Play');
    play.setAttribute('aria-label', `Play ${title.toLowerCase()}`);
    play.append(icon, playLabel);
    const stop = button('pd-button-secondary', 'Stop replay');
    stop.hidden = true;
    const retryPoster = button('pd-button-secondary', 'Retry preview');
    retryPoster.hidden = true;
    const footer = element('div', 'pd-player-footer');
    const feedback = element('p', 'pd-feedback', '');
    feedback.hidden = true;
    feedback.setAttribute('role', 'status');
    feedback.setAttribute('aria-live', 'polite');
    const actions = element('div', 'pd-player-actions');
    actions.append(retryPoster, stop);
    footer.append(feedback, actions);
    footer.hidden = true;
    frame.append(placeholder, poster, play);
    container.append(frame, footer);

    let disposed = false;
    let currentMedia = null;
    let posterFailed = false;
    let replayFailed = false;
    const posterURL = media.poster ? assetURL(media.poster, manifestURL, publicMediaBase) : null;

    function notify(message, failed = false, visuallyHidden = false) {
      feedback.textContent = message;
      feedback.hidden = !message;
      feedback.classList.toggle('pd-feedback-error', failed);
      feedback.classList.toggle('pd-sr-only', visuallyHidden);
      footer.hidden = (!message || visuallyHidden) && stop.hidden && retryPoster.hidden;
    }

    function setLoading(isLoading) {
      play.disabled = isLoading;
      play.setAttribute('aria-busy', String(isLoading));
      if (isLoading) {
        play.hidden = false;
        playLabel.textContent = 'Loading replay...';
        play.setAttribute('aria-label', `Loading ${title.toLowerCase()}`);
      }
    }

    function unloadMedia() {
      if (!currentMedia) return;
      const node = currentMedia;
      currentMedia = null;
      node.onload = null;
      node.onerror = null;
      node.onloadeddata = null;
      node.onended = null;
      if (media.kind === 'video') node.pause();
      node.removeAttribute('src');
      if (media.kind === 'video') node.load();
      node.remove();
    }

    function restorePoster() {
      unloadMedia();
      setLoading(false);
      poster.hidden = !posterURL || posterFailed;
      placeholder.hidden = !posterURL || !posterFailed;
      play.hidden = false;
      stop.hidden = true;
      retryPoster.hidden = !posterURL || !posterFailed;
      replayFailed = false;
      playLabel.textContent = 'Play';
      play.setAttribute('aria-label', `Play ${title.toLowerCase()}`);
    }

    function mediaError(node) {
      if (disposed || currentMedia !== node) return;
      const moveFocus = container.contains(document.activeElement);
      restorePoster();
      replayFailed = true;
      playLabel.textContent = 'Retry replay';
      play.setAttribute('aria-label', 'Retry replay');
      notify('Could not load replay.', true);
      if (moveFocus) play.focus();
    }

    poster.onload = () => {
      if (disposed) return;
      const retryHadFocus = document.activeElement === retryPoster;
      posterFailed = false;
      placeholder.hidden = true;
      retryPoster.hidden = true;
      if (!currentMedia) {
        notify(replayFailed
          ? 'Could not load replay.'
          : '', replayFailed);
      }
      if (retryHadFocus) (currentMedia ? stop : play).focus();
    };
    poster.onerror = () => {
      if (disposed) return;
      posterFailed = true;
      poster.hidden = true;
      placeholder.hidden = false;
      placeholder.textContent = 'Still preview unavailable';
      retryPoster.hidden = Boolean(currentMedia);
      if (!currentMedia) notify('Could not load preview.', true);
    };
    if (posterURL) {
      poster.src = posterURL;
    } else {
      poster.hidden = true;
      placeholder.hidden = true;
    }

    retryPoster.addEventListener('click', () => {
      poster.hidden = Boolean(currentMedia);
      notify('Loading the still preview...');
      poster.removeAttribute('src');
      poster.src = posterURL;
    });

    function stopPlayback(moveFocus = false) {
      if (disposed) return;
      restorePoster();
      notify(posterFailed ? 'Could not load preview.' : '', posterFailed);
      if (moveFocus) play.focus();
    }
    stop.addEventListener('click', () => stopPlayback(true));

    function startPlayback(moveFocus = false) {
      if (disposed) return;
      if (currentMedia) {
        if (media.kind !== 'video' || !currentMedia.ended) return;
        unloadMedia();
      }
      const node = element(media.kind === 'gif' ? 'img' : 'video', 'pd-media');
      currentMedia = node;
      replayFailed = false;
      node.hidden = true;
      const onReady = () => {
        if (disposed || currentMedia !== node) return;
        setLoading(false);
        play.hidden = true;
        node.hidden = false;
        poster.hidden = true;
        placeholder.hidden = true;
        notify('');
      };
      node.onerror = () => mediaError(node);
      if (media.kind === 'gif') {
        node.alt = `${title} of ${caseTitle}`;
        node.onload = onReady;
      } else {
        node.controls = nativeControls;
        node.muted = true;
        node.loop = false;
        node.preload = 'none';
        node.playsInline = true;
        if (posterURL) node.poster = posterURL;
        node.setAttribute('aria-label', `${title} of ${caseTitle}`);
        node.onloadeddata = onReady;
        node.onended = () => {
          if (disposed || currentMedia !== node) return;
          if (nativeControls) {
            notify('');
            return;
          }
          const stopHadFocus = document.activeElement === stop;
          playLabel.textContent = 'Replay';
          play.setAttribute('aria-label', `Replay ${title.toLowerCase()}`);
          play.hidden = false;
          stop.hidden = true;
          notify('');
          if (stopHadFocus) play.focus();
        };
        node.hidden = false;
      }
      frame.insertBefore(node, play);
      setLoading(true);
      stop.hidden = nativeControls;
      retryPoster.hidden = true;
      notify('Loading replay...', false, true);
      if (moveFocus) (nativeControls ? node : stop).focus();
      // Viewing a feature starts playback; the page itself never preloads recordings.
      node.src = replayURL;
      if (media.kind === 'video') {
        node.play().catch((error) => {
          if (disposed || currentMedia !== node) return;
          if (error.name === 'NotAllowedError') {
            setLoading(false);
            play.hidden = true;
            node.hidden = false;
            node.controls = true;
            notify('Press play to start.');
          } else {
            mediaError(node);
          }
        });
      }
    }
    play.addEventListener('click', () => startPlayback(true));

    return {
      element: container,
      start: startPlayback,
      stop: stopPlayback,
      destroy() {
        disposed = true;
        unloadMedia();
        poster.onload = null;
        poster.onerror = null;
        poster.removeAttribute('src');
      }
    };
  }

  function renderDiff(diff, title = 'Code removed') {
    const box = element('div', 'pd-diff-box');
    const heading = element('div', 'pd-diff-heading');
    heading.append(element('strong', '', title), element('span', '', '- removed / + replacement'));
    const pre = element('pre', 'pd-diff');
    pre.tabIndex = 0;
    pre.setAttribute('role', 'region');
    pre.setAttribute('aria-label', 'Masking diff. Minus marks removals; plus marks additions. Scroll to read more.');
    const code = element('code', 'pd-diff-code');
    const lines = diff.split('\n').filter((line) => !line.startsWith('index '));
    lines.forEach((line, index) => {
      let kind = '';
      if (/^(diff |index |--- |\+\+\+ )/.test(line)) kind = 'file';
      else if (line.startsWith('@@')) kind = 'hunk';
      else if (line.startsWith('-')) kind = 'remove';
      else if (line.startsWith('+')) kind = 'add';
      code.append(element('span', `pd-diff-line${kind ? ` pd-diff-${kind}` : ''}`, line));
      if (index < lines.length - 1) code.append(document.createTextNode('\n'));
    });
    pre.append(code);
    box.append(heading, pre);
    return box;
  }

  let codeBrowserId = 0;

  function codeBrowser(files, selectedPath, onSelect = () => {}) {
    const box = element('div', 'pd-code-browser');
    if (!files.length) {
      box.append(element('p', 'pd-note', 'No changed files are available for this mask.'));
      return box;
    }
    const prefix = `pd-code-${++codeBrowserId}`;
    const sidebar = element('div', 'pd-code-sidebar');
    sidebar.append(element('p', 'pd-label', `Changed files (${files.length})`));
    const list = element('div', 'pd-code-files');
    list.setAttribute('role', 'tablist');
    list.setAttribute('aria-label', 'Changed implementation files');
    list.setAttribute('aria-orientation', 'vertical');
    const code = element('div', 'pd-code-content');
    code.id = `${prefix}-diff`;
    code.setAttribute('role', 'tabpanel');
    code.tabIndex = 0;
    const controls = files.map((file, index) => {
      const control = button('pd-code-file', '');
      const slash = file.path.lastIndexOf('/');
      control.append(element('strong', '', file.path.slice(slash + 1)));
      if (slash >= 0) control.append(element('span', 'pd-code-directory', file.path.slice(0, slash + 1)));
      control.title = file.path;
      control.dataset.path = file.path;
      control.id = `${prefix}-file-${index}`;
      control.setAttribute('role', 'tab');
      control.setAttribute('aria-controls', code.id);
      control.addEventListener('click', () => render(index));
      control.addEventListener('keydown', (event) => {
        let next;
        if (event.key === 'ArrowDown') next = (index + 1) % files.length;
        else if (event.key === 'ArrowUp') next = (index + files.length - 1) % files.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = files.length - 1;
        else return;
        event.preventDefault();
        render(next);
        controls[next].focus({ preventScroll: true });
        controls[next].scrollIntoView({ block: 'nearest' });
      });
      list.append(control);
      return control;
    });
    function render(index) {
      const file = files[index];
      controls.forEach((control, position) => {
        control.setAttribute('aria-selected', String(index === position));
        control.tabIndex = index === position ? 0 : -1;
      });
      code.setAttribute('aria-labelledby', controls[index].id);
      code.replaceChildren(...(file.parts
        ? file.parts.map((part) => renderDiff(part.diff, part.feature))
        : [renderDiff(file.diff)]));
      code.scrollTop = 0;
      onSelect(file.path);
    }
    sidebar.append(list);
    box.append(sidebar, code);
    render(Math.max(0, files.findIndex((file) => file.path === selectedPath)));
    return box;
  }

  function combineSelection(item, selected) {
    const components = item.composition.components.filter((component) => selected.has(component.trace));
    requireData(components.length > 0 && components.length === selected.size, 'select valid code removals before combining');
    const recorded = item.craft.examples.find((example) =>
      example.traces.length === selected.size && example.traces.every((name) => selected.has(name)));
    if (recorded) return { ...recorded, draft: false };
    if (components.length === 1) {
      const component = components[0];
      return {
        id: component.id, depth: 1, traces: [component.trace], draft: false,
        brief: component.brief, diffFiles: component.diffFiles,
        targetFile: component.diffFiles[0].path, removedLines: component.removedLines,
        filesChanged: component.files
      };
    }
    const files = new Map();
    components.forEach((component) => {
      const node = item.graph.nodes.find((entry) => entry.id === component.trace);
      component.diffFiles.forEach((file) => {
        if (!files.has(file.path)) files.set(file.path, { path: file.path, parts: [] });
        files.get(file.path).parts.push({ feature: behaviorLabel(node), diff: file.diff });
      });
    });
    return {
      id: null, depth: components.length, draft: true,
      traces: components.map((component) => component.trace),
      diffFiles: [...files.values()],
      targetFile: components[components.length - 1].diffFiles[0].path,
      brief: 'Restore the following features to match the working reference.\n\n' +
        components.map((component, index) =>
          `Restoration stage ${index + 1} - ${component.trace}:\n${component.brief}`).join('\n\n') +
        '\n\nPreserve the prerequisite behavior needed to reach these features.'
    };
  }

  function taskBrief(text) {
    const box = element('div', 'pd-generated-brief');
    box.tabIndex = 0;
    box.setAttribute('role', 'region');
    box.setAttribute('aria-label', 'Generated task brief');
    text.split(/\n\s*\n/).forEach((paragraph) => {
      const stage = paragraph.match(/^Restoration stage \d+ - ([\w]+):\n([\s\S]*)$/);
      const checkpoint = paragraph.match(/^Validation checkpoint \d+ - ([\w]+) \(not masked; preserve existing behavior\):\n([\s\S]*)$/);
      if (stage || checkpoint) {
        const match = stage || checkpoint;
        const label = behaviorLabel({ name: match[1], label: match[1].replaceAll('_', ' ') });
        box.append(element('h5', 'pd-brief-feature', checkpoint ? `${label} (unchanged)` : label),
          element('p', '', match[2]));
      } else {
        box.append(element('p', '', paragraph));
      }
    });
    return box;
  }

  function renderChecks(checks) {
    const section = element('div', 'pd-verification');
    section.append(element('span', 'pd-label', 'Mask verification'));
    if (checks.length === 0) {
      section.append(element('p', 'pd-note', 'No verification checks were provided for this case.'));
      return section;
    }
    const list = element('ul', 'pd-checks');
    checks.forEach((check) => {
      const row = element('li');
      const heading = element('div', 'pd-check-heading');
      heading.append(element('span', '', check.label), status(CHECK_NAMES[check.status], check.status));
      row.append(heading);
      if (check.detail) row.append(element('p', 'pd-check-detail', check.detail));
      list.append(row);
    });
    section.append(list);
    return section;
  }

  function svgElement(tag, attributes, text) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function createWorkflow(item, onOpen) {
    const list = element('ul', 'pd-workflow pd-feature-grid');
    list.setAttribute('aria-label', 'Mined features available in this example');
    item.lineage.filter((entry) => entry.masked).forEach((entry) => {
      const node = item.graph.nodes.find((candidate) => candidate.id === entry.name);
      const row = element('li');
      const control = button('pd-workflow-step', '');
      control.dataset.node = node.id;
      control.setAttribute('aria-haspopup', 'dialog');
      control.setAttribute('aria-label', `Explore ${behaviorLabel(node)}`);
      control.append(
        element('span', 'pd-workflow-name', behaviorLabel(node)),
        element('span', 'pd-workflow-action', 'View')
      );
      control.addEventListener('click', () => onOpen(node, control));
      row.append(control);
      list.append(row);
    });
    return list;
  }

  function createGraph(item, onOpen) {
    const box = element('div', 'pd-graph');
    const controls = element('div', 'pd-graph-controls');
    const layerLabel = element('label', 'pd-layer-control');
    const layerText = element('span', 'pd-label');
    const range = element('input', 'pd-layer-range');
    const maxDepth = Math.max(...item.graph.nodes.map((node) => node.depth));
    range.type = 'range';
    range.min = '1';
    range.max = String(maxDepth);
    range.value = range.max;
    range.setAttribute('aria-label', 'Reveal prerequisite layers');
    layerLabel.append(layerText, range);
    const filterLabel = element('label', 'pd-graph-filter');
    const filter = element('input');
    filter.type = 'checkbox';
    filterLabel.append(filter, document.createTextNode('Only the example workflow'));
    controls.append(layerLabel, filterLabel);
    const scroll = element('div', 'pd-graph-scroll');
    const summary = element('p', 'pd-note pd-graph-summary');
    const legend = element('p', 'pd-note pd-graph-legend',
      'Numbered dots follow the example workflow. Smaller dots are other behaviors discovered in this app. Every dot opens a behavior card.');
    box.append(controls, scroll, summary, legend,
      element('p', 'pd-note', 'The slider reveals dependency layers, not discovery time. Arrow keys move between dots; Enter opens a card.'));
    let selectedId = item.graph.target;
    let drawing;

    function selectNode(id, open = false) {
      selectedId = id;
      let selectedControl;
      drawing.querySelectorAll('[data-node]').forEach((node) => {
        const selected = node.dataset.node === id;
        node.classList.toggle('pd-node-selected', selected);
        node.setAttribute('tabindex', selected ? '0' : '-1');
        if (selected) selectedControl = node;
      });
      if (open) onOpen(item.graph.nodes.find((node) => node.id === id), selectedControl);
    }

    function draw() {
      const limit = Number(range.value);
      const visible = item.graph.nodes.filter((node) => node.depth <= limit && (!filter.checked || node.onPath))
        .sort((a, b) => a.depth - b.depth || Number(b.onPath) - Number(a.onPath) ||
          (a.parent || '').localeCompare(b.parent || '') || a.id.localeCompare(b.id));
      const counts = new Map();
      const positions = new Map();
      visible.forEach((node) => {
        const row = counts.get(node.depth) || 0;
        positions.set(node.id, { x: 42 + (node.depth - 1) * 76, y: 65 + row * 34 });
        counts.set(node.depth, row + 1);
      });
      const width = Math.max(720, 84 + (maxDepth - 1) * 76);
      const height = Math.max(150, 98 + Math.max(...counts.values()) * 34);
      drawing = svgElement('svg', {
        viewBox: `0 0 ${width} ${height}`, width, height, class: 'pd-tree-svg',
        role: 'group', 'aria-label': 'Replay-verified prerequisite graph'
      });
      for (let depth = 1; depth <= maxDepth; depth += 1) {
        const x = 42 + (depth - 1) * 76;
        drawing.append(svgElement('text', { x, y: 23, class: 'pd-layer-number', 'text-anchor': 'middle' }, depth));
      }
      visible.forEach((node) => {
        if (!node.parent) return;
        const from = positions.get(node.parent);
        const to = positions.get(node.id);
        const midpoint = (from.x + to.x) / 2;
        drawing.append(svgElement('path', {
          d: `M ${from.x} ${from.y} C ${midpoint} ${from.y}, ${midpoint} ${to.y}, ${to.x} ${to.y}`,
          class: `pd-tree-edge${node.onPath ? ' pd-tree-edge-path' : ''}`
        }));
      });
      visible.forEach((node, position) => {
        const point = positions.get(node.id);
        const order = item.lineage.findIndex((entry) => entry.name === node.id);
        const role = node.onPath ? 'on the example workflow' : 'another discovered behavior';
        const group = svgElement('g', {
          transform: `translate(${point.x}, ${point.y})`,
          class: `pd-tree-node${node.onPath ? ' pd-node-path' : ''}${node.onPath && !node.masked ? ' pd-node-setup' : ''}`,
          role: 'button', 'aria-haspopup': 'dialog',
          'aria-label': `Explore ${behaviorLabel(node)}; ${role}`,
          'data-node': node.id
        });
        group.append(
          svgElement('title', {}, `Open: ${behaviorLabel(node)}`),
          svgElement('circle', { r: 17, class: 'pd-node-hit' }),
          svgElement('circle', { r: node.onPath ? 15 : 7, class: 'pd-node-dot' })
        );
        if (node.onPath) group.append(svgElement('text', { 'text-anchor': 'middle', y: 4 }, order + 1));
        group.addEventListener('click', () => selectNode(node.id, true));
        group.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            selectNode(node.id, true);
            return;
          }
          let next;
          if (['ArrowRight', 'ArrowDown'].includes(event.key)) next = (position + 1) % visible.length;
          else if (['ArrowLeft', 'ArrowUp'].includes(event.key)) next = (position - 1 + visible.length) % visible.length;
          else if (event.key === 'Home') next = 0;
          else if (event.key === 'End') next = visible.length - 1;
          else return;
          event.preventDefault();
          selectNode(visible[next].id);
          const focused = Array.from(drawing.querySelectorAll('[data-node]')).find((entry) => entry.dataset.node === selectedId);
          focused.focus();
        });
        drawing.append(group);
      });
      scroll.replaceChildren(drawing);
      if (!positions.has(selectedId)) {
        const path = visible.filter((node) => node.onPath);
        selectedId = path[path.length - 1].id;
      }
      layerText.textContent = `Prerequisite layers 1-${limit} / ${maxDepth}`;
      summary.textContent = `${visible.length} of ${item.graph.nodes.length} replay-verified behaviors shown. Each edge means the earlier behavior is needed to reach the next.`;
      selectNode(selectedId);
    }
    range.addEventListener('input', draw);
    filter.addEventListener('change', draw);
    draw();
    return box;
  }

  function createBuilder(item, state, onCombine, onBuild, onOpenMask) {
    const box = element('div', 'pd-builder');
    const list = element('div', 'pd-removal-inputs');
    const result = element('div', 'pd-combine-action');
    const combine = button('', 'Combine masks');
    combine.setAttribute('aria-haspopup', 'dialog');
    const build = button('', 'Build task');
    build.setAttribute('aria-haspopup', 'dialog');
    result.append(combine, build);
    box.append(list, result);

    function update() {
      result.hidden = state.removed.size === 0;
      combine.disabled = state.selected.size === 0;
      combine.textContent = state.combined ? 'View combined mask' : 'Combine masks';
      build.hidden = !state.combined;
    }
    function renderCard(card, component) {
      const node = item.graph.nodes.find((entry) => entry.id === component.trace);
      card.replaceChildren();
      card.className = 'pd-removal-card';
      card.dataset.trace = component.trace;
      card.dataset.state = state.removed.has(component.trace) ? 'mask' : 'feature';
      if (!state.removed.has(component.trace)) {
        card.classList.add('pd-craft-feature');
        const remove = button('pd-button-secondary pd-create-mask', "Remove this feature's code");
        remove.addEventListener('click', () => {
          state.removed.add(component.trace);
          state.selected.add(component.trace);
          state.combined = null;
          renderCard(card, component);
          update();
          card.querySelector('input').focus({ preventScroll: true });
        });
        card.append(element('strong', '', behaviorLabel(node)), remove);
        return;
      }
      const label = element('label', 'pd-mask-choice');
      const checkbox = element('input');
      checkbox.type = 'checkbox';
      checkbox.checked = state.selected.has(component.trace);
      checkbox.dataset.trace = component.trace;
      label.append(checkbox, element('strong', '', behaviorLabel(node)));
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) state.selected.add(component.trace);
        else state.selected.delete(component.trace);
        state.combined = null;
        update();
      });
      const deleted = component.diffFiles[0].diff.split('\n')
        .filter((line) => line.startsWith('-') && !line.startsWith('---') && line.slice(1).trim())
        .slice(0, 3).join('\n');
      const inspect = button('pd-inspect-mask', 'Inspect code removal');
      inspect.setAttribute('aria-haspopup', 'dialog');
      inspect.addEventListener('click', () => onOpenMask(component, inspect));
      card.append(label, element('code', 'pd-removal-teaser', deleted), inspect);
    }
    item.composition.components.forEach((component) => {
      const card = element('div');
      renderCard(card, component);
      list.append(card);
    });
    combine.addEventListener('click', () => {
      state.combined = combineSelection(item, state.selected);
      update();
      onCombine(state.combined, combine);
    });
    build.addEventListener('click', () => onBuild(state.combined, build));
    update();
    return box;
  }

  function initialize(root, index) {
    if (root.dataset.pdInitialized) return;
    root.dataset.pdInitialized = 'true';
    const prefix = `pd-explorer-${index + 1}`;
    const find = (selector) => root.querySelector(selector);
    const caseOptions = find('[data-pd-case]');
    const caseLabel = find('[data-pd-case-label]');
    const notice = find('[data-pd-notice]');
    const message = find('[data-pd-message]');
    const retry = find('[data-pd-retry]');
    const content = find('[data-pd-content]');
    const live = find('[data-pd-live]');
    const tabs = Array.from(root.querySelectorAll('[data-pd-stage]'));
    const panels = Array.from(root.querySelectorAll('[data-pd-panel]'));
    let manifest = null;
    let manifestURL = null;
    let activeStage = 'mine';
    let selectedCaseIndex = 0;
    let player = null;
    let loading = false;
    const builderStates = new Map();
    const selectedFiles = new Map();
    const dialog = createDialog(root, prefix);
    initializeTraceExamples(root, prefix);
    initializeFigureZoom(root, prefix);
    const dashboardSoon = document.querySelector(`[data-pd-dashboard-soon="${root.id}"]`);
    if (dashboardSoon) {
      dashboardSoon.addEventListener('click', () => {
        dialog.open({
          title: 'Full dashboard coming soon',
          category: '',
          body: element('p', 'pd-description', "We're getting it ready. Please check back soon!"),
          trigger: dashboardSoon
        });
      });
    }

    function builderState(item) {
      if (!builderStates.has(item.id)) builderStates.set(item.id, { removed: new Set(), selected: new Set(), combined: null });
      return builderStates.get(item.id);
    }

    caseOptions.id = `${prefix}-cases`;
    caseLabel.id = `${prefix}-case-label`;
    caseOptions.setAttribute('aria-labelledby', caseLabel.id);
    tabs.forEach((tab, tabIndex) => {
      const panel = panels[tabIndex];
      tab.id = `${prefix}-${tab.dataset.pdStage}-tab`;
      panel.id = `${prefix}-${tab.dataset.pdStage}-panel`;
      tab.setAttribute('aria-controls', panel.id);
      panel.setAttribute('aria-labelledby', tab.id);
    });

    function goToStage(stage) {
      dialog.close(false);
      activeStage = stage;
      renderStage();
      const tab = tabs[STAGES.indexOf(stage)];
      tab.focus({ preventScroll: true });
      root.scrollIntoView({ block: 'start' });
    }

    function openBehavior(item, node, trigger) {
      const body = element('div', 'pd-behavior-card');
      body.dataset.behavior = node.id;
      const observed = node.id === 'manual_login'
        ? 'The user signs in through the login form and reaches the authenticated app.'
        : node.id === 'manual_signup'
          ? 'A new user registers an account and reaches the authenticated app.'
          : node.description;
      body.append(element('p', 'pd-dialog-lead', observed));
      const parent = item.graph.nodes.find((entry) => entry.id === node.parent);
      const prerequisite = element('p', 'pd-description');
      prerequisite.append(element('strong', '', 'Prerequisite: '),
        document.createTextNode(parent ? behaviorLabel(parent) : 'no earlier mined behavior.'));
      body.append(prerequisite);
      let replay = null;
      if (node.media) {
        replay = createPlayer(node.media, manifestURL, 'Working app', behaviorLabel(node));
        body.append(replay.element);
      } else {
        body.append(element('p', 'pd-note', 'A replay for this behavior is not included in the inline example.'));
      }
      dialog.open({
        title: behaviorLabel(node),
        category: appLabel(item),
        body, trigger,
        cleanup: () => { if (replay) replay.destroy(); }
      });
      if (replay) replay.start();
    }

    function openMask(item, component, trigger) {
      const node = item.graph.nodes.find((entry) => entry.id === component.trace);
      const body = element('div', 'pd-atomic-code');
      body.dataset.trace = component.trace;
      body.append(element('p', 'pd-dialog-lead',
        'This is the implementation removal that turns the working feature into a repair task.'),
      browseCode(component.diffFiles, `${item.id}:mask:${component.id}`));
      openCodeDialog({ title: `${behaviorLabel(node)}: code removed`, category: '', body, trigger });
    }

    function browseCode(files, key, targetFile) {
      return codeBrowser(files, selectedFiles.get(key) || targetFile,
        (path) => selectedFiles.set(key, path));
    }

    function openCodeDialog(options) {
      dialog.open(options);
      options.body.querySelector('.pd-code-file[aria-selected="true"]')
        ?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }

    function openCombined(item, example, trigger) {
      const body = element('div', 'pd-combined-mask');
      body.dataset.restorationDepth = String(example.depth);
      if (example.draft) body.append(element('p', 'pd-description',
        'Draft combination: edits to shared files are shown per feature and may need conflict resolution.'));
      else if (example.depth > 1) body.append(element('p', 'pd-description',
        `${example.depth} feature masks combined into one patch using ${example.method === 'Agent-assisted merge' ? 'an agent-assisted' : 'a deterministic'} merge.`));
      body.append(browseCode(example.diffFiles, `${item.id}:combined:${example.traces.join(',')}`, example.targetFile));
      const build = button('pd-behavior-add', 'Build task');
      build.addEventListener('click', () => openTask(item, example, build));
      body.append(build);
      openCodeDialog({ title: example.draft ? 'Mask combination (draft)' : example.depth > 1 ? 'Combined mask' : 'Code mask', category: '', body, trigger });
    }

    function openTask(item, example, trigger) {
      const body = element('div', 'pd-task-preview');
      body.dataset.restorationDepth = String(example.depth);
      body.dataset.taskId = example.id || '';
      body.dataset.draft = String(example.draft);
      body.append(taskBrief(example.brief));
      if (example.id === item.id) {
        const watch = button('pd-behavior-add', 'Watch the repair of this task');
        watch.addEventListener('click', () => goToStage('patch'));
        body.append(watch);
      }
      dialog.open({
        title: `${example.draft ? 'Task draft' : 'Task'}: rebuild ${example.depth} ${example.depth === 1 ? 'feature' : 'features'}`,
        category: '',
        body, trigger
      });
    }

    function renderStage(announce = true) {
      const item = manifest.cases[selectedCaseIndex];
      find('[data-pd-app-source]').replaceChildren(createApplicationSource(item.application));
      dialog.close(false);
      if (player) {
        player.destroy();
        player = null;
      }
      panels.forEach((panel) => {
        panel.replaceChildren();
        panel.hidden = panel.dataset.pdPanel !== activeStage;
      });
      tabs.forEach((tab) => {
        const selected = tab.dataset.pdStage === activeStage;
        tab.setAttribute('aria-selected', String(selected));
        tab.tabIndex = selected ? 0 : -1;
      });
      const panel = panels[STAGES.indexOf(activeStage)];
      let description;
      if (activeStage === 'mine') {
        description = element('p', 'pd-description',
          'Select a mined feature to watch its recorded workflow.');
        const workflow = createWorkflow(item, (node, trigger) => openBehavior(item, node, trigger));
        panel.append(description, workflow);
      } else if (activeStage === 'craft') {
        description = element('p', 'pd-description',
          'Remove a mined feature\'s code to create its mask. Select the masks to combine, then build a task.');
        const builder = createBuilder(item, builderState(item),
          (example, trigger) => openCombined(item, example, trigger),
          (example, trigger) => openTask(item, example, trigger),
          (component, trigger) => openMask(item, component, trigger));
        panel.append(description, builder);
      } else {
        description = element('p', 'pd-description', item.patch.binarySuccess
          ? `${item.patch.model} restores the complete ${item.depth}-feature task. Watch how it connects the reference behavior to code changes.`
          : `${item.patch.model} attempts the complete ${item.depth}-feature task, but does not restore the whole workflow.`);
        player = createPlayer(item.patch.media, manifestURL, 'Agent replay', item.title, null, { controls: true });
        panel.append(description, player.element);
      }
      if (announce) live.textContent = `${item.title}. ${STAGE_NAMES[activeStage]} stage.`;
    }

    function selectCase(index, announce = true) {
      selectedCaseIndex = index;
      Array.from(caseOptions.children).forEach((option, position) => {
        const selected = position === selectedCaseIndex;
        option.setAttribute('aria-selected', String(selected));
        option.tabIndex = selected ? 0 : -1;
      });
      renderStage(announce);
    }

    function createCaseOption(item, caseIndex) {
      const option = element('button', 'pd-case-option');
      option.type = 'button';
      option.setAttribute('role', 'tab');
      option.append(
        element('strong', 'pd-case-option-app', item.app),
        element('span', 'pd-case-option-meta', `${item.patch.model} · depth ${item.depth}`)
      );
      option.addEventListener('click', () => {
        if (caseIndex !== selectedCaseIndex) selectCase(caseIndex);
      });
      option.addEventListener('keydown', (event) => {
        let next;
        if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (caseIndex + 1) % manifest.cases.length;
        else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (caseIndex + manifest.cases.length - 1) % manifest.cases.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = manifest.cases.length - 1;
        else return;
        event.preventDefault();
        selectCase(next);
        caseOptions.children[next].focus();
      });
      return option;
    }

    tabs.forEach((tab, tabIndex) => {
      tab.addEventListener('click', () => {
        if (!manifest || tab.dataset.pdStage === activeStage) return;
        activeStage = tab.dataset.pdStage;
        renderStage();
      });
      tab.addEventListener('keydown', (event) => {
        let next;
        if (event.key === 'ArrowRight') next = (tabIndex + 1) % tabs.length;
        else if (event.key === 'ArrowLeft') next = (tabIndex + tabs.length - 1) % tabs.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = tabs.length - 1;
        else return;
        event.preventDefault();
        tabs[next].focus();
        tabs[next].click();
      });
    });

    async function loadManifest() {
      if (loading) return;
      const retryHadFocus = document.activeElement === retry;
      loading = true;
      content.hidden = true;
      notice.hidden = false;
      notice.classList.remove('pd-notice-error');
      retry.hidden = true;
      message.textContent = 'Loading interactive examples...';
      root.setAttribute('aria-busy', 'true');
      try {
        const source = root.dataset.src;
        requireData(isLabel(source), 'a manifest URL is required');
        const requestedURL = new URL(source, document.baseURI);
        requireData(
          ['http:', 'https:'].includes(requestedURL.protocol) && requestedURL.origin === window.location.origin,
          'the manifest must be served by this site'
        );
        const [response, sources] = await Promise.all([
          fetch(requestedURL.href, {
            headers: { Accept: 'application/json' },
            credentials: 'same-origin',
            mode: 'cors',
            redirect: 'error'
          }),
          loadApplicationSources(root.dataset.applicationSourcesSrc)
        ]);
        if (!response.ok) throw new Error(`The case manifest returned HTTP ${response.status}.`);
        manifestURL = new URL(response.url || requestedURL.href);
        requireData(manifestURL.origin === requestedURL.origin, 'the manifest redirected to another site');
        manifest = validateManifest(await response.json(), manifestURL);
        caseOptions.replaceChildren();
        manifest.cases.forEach((item, caseIndex) => {
          requireData(sources.has(item.appId), `missing application source for ${item.appId}`);
          item.application = sources.get(item.appId);
          caseOptions.append(createCaseOption(item, caseIndex));
        });
        selectedCaseIndex = 0;
        selectCase(0, false);
        notice.hidden = true;
        content.hidden = false;
        live.textContent = `${manifest.cases.length} cases loaded. ${manifest.cases[0].title}. ${STAGE_NAMES[activeStage]} stage.`;
        if (retryHadFocus) caseOptions.children[0].focus();
      } catch (error) {
        manifest = null;
        const reason = error instanceof SyntaxError
          ? 'The case manifest is not valid JSON.'
          : error instanceof Error ? error.message : 'An unexpected loading error occurred.';
        message.textContent = `The explorer could not load. ${reason} Retry to request the examples again.`;
        notice.classList.add('pd-notice-error');
        retry.hidden = false;
        if (retryHadFocus) retry.focus();
      } finally {
        loading = false;
        root.removeAttribute('aria-busy');
      }
    }

    retry.addEventListener('click', loadManifest);
    loadManifest();
  }

  window.ProgramDistillUI = Object.freeze({
    element, button, status, stats, createDialog, createPlayer, createGraph, behaviorLabel, renderDiff, combineSelection,
    loadApplicationSources, createApplicationSource
  });
  document.querySelectorAll('[data-pd-explorer]').forEach(initialize);
})();
