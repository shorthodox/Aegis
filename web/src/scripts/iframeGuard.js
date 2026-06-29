// ============================================================
// iframeGuard — Safe third-party script & iframe loading
// ============================================================

const LOAD_TIMEOUT_MS = 8000;
const _loaded = new Set();

/**
 * Dynamically load a third-party script, guarded with timeout and error events.
 * Safe to call multiple times for the same URL — deduplicates automatically.
 */
export function loadThirdPartyScript(src, { timeout = LOAD_TIMEOUT_MS } = {}) {
  if (_loaded.has(src)) return Promise.resolve();

  // If a script tag already exists but is still loading, wait for it instead
  // of resolving immediately (which would leave window.Razorpay undefined).
  const existing = document.querySelector(`script[src="${src}"]`);
  if (existing) {
    if (existing.dataset.loaded === 'true') {
      _loaded.add(src);
      return Promise.resolve();
    }
    // Script tag exists but hasn't fired load yet — wait for it
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Script load timeout (existing tag): ${src}`));
      }, timeout);
      existing.addEventListener('load', () => { clearTimeout(timer); _loaded.add(src); resolve(); });
      existing.addEventListener('error', () => { clearTimeout(timer); reject(new Error(`Script load failed (existing tag): ${src}`)); });
    });
  }

  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;

    const timer = setTimeout(() => {
      script.remove();
      reject(new Error(`Script load timeout: ${src}`));
    }, timeout);

    script.addEventListener('load', () => {
      clearTimeout(timer);
      script.dataset.loaded = 'true';
      _loaded.add(src);
      resolve();
    });

    script.addEventListener('error', () => {
      clearTimeout(timer);
      script.remove();
      reject(new Error(`Script load failed: ${src}`));
    });

    document.head.appendChild(script);
  });
}

// ============================================================
// Sandbox attribute presets
// ============================================================
//
// Usage: iframe.setAttribute('sandbox', SANDBOX.payment);
//
// payment — payment modals that need forms, popups, scripts,
//            same-origin comms, and user-activated navigation.
//
// embed   — read-only widgets (charts, maps): can run scripts
//            but cannot reach the parent page's DOM or storage.
//
// strict  — maximum isolation; nothing is permitted.
//            Use for fully untrusted display-only content.
//
export const SANDBOX = {
  payment: [
    'allow-forms',
    'allow-popups',
    'allow-popups-to-escape-sandbox', // lets the payment window escape to top-level
    'allow-scripts',
    'allow-same-origin',              // needed for postMessage callbacks
    'allow-top-navigation-by-user-activation',
  ].join(' '),
  embed: 'allow-scripts allow-same-origin',
  strict: '',
};

// ============================================================
// createGuardedIframe
// ============================================================
/**
 * Create an iframe, append it to `container`, and detect load failures.
 *
 * Fails if:
 *   - container is not in the DOM
 *   - the load event fires with a chrome-error:// / about:error URL
 *   - the load/error/timeout fires before the frame finishes loading
 *
 * @param {string}      src        URL to load
 * @param {HTMLElement} container  DOM node to append the iframe to
 * @param {object}      [options]
 * @param {string|null} [options.sandbox]  Sandbox string or null to omit the attribute
 * @param {number}      [options.timeout]  Milliseconds before a timeout error (default 8000)
 * @param {function}    [options.onError]  Called with the Error on failure
 * @param {function}    [options.onLoad]   Called with the iframe element on success
 * @param {...*}        [options.attrs]    Any remaining keys are set as iframe attributes
 * @returns {Promise<HTMLIFrameElement>}
 */
export function createGuardedIframe(src, container, {
  sandbox = SANDBOX.embed,
  timeout = LOAD_TIMEOUT_MS,
  onError,
  onLoad,
  ...attrs
} = {}) {
  return new Promise((resolve, reject) => {
    if (!container || !document.contains(container)) {
      const err = new Error(`IframeGuard: container not in DOM for "${src}"`);
      onError?.(err);
      return reject(err);
    }

    const iframe = document.createElement('iframe');
    iframe.src = src;
    if (sandbox !== null) iframe.setAttribute('sandbox', sandbox);
    Object.entries(attrs).forEach(([k, v]) => iframe.setAttribute(k, String(v)));

    const timer = setTimeout(() => {
      iframe.remove();
      const err = new Error(`IframeGuard: load timeout for "${src}"`);
      onError?.(err);
      reject(err);
    }, timeout);

    iframe.addEventListener('load', () => {
      clearTimeout(timer);
      let ok = true;
      try {
        const loc = iframe.contentWindow?.location?.href ?? '';
        // Chrome renders chrome-error:// when the resource fails to load
        if (loc.startsWith('chrome-error://') || loc.startsWith('about:error')) ok = false;
      } catch {
        // SecurityError: frame is cross-origin/sandboxed — load itself succeeded
      }
      if (!ok) {
        iframe.remove();
        const err = new Error(`IframeGuard: browser error page for "${src}"`);
        onError?.(err);
        return reject(err);
      }
      onLoad?.(iframe);
      resolve(iframe);
    });

    iframe.addEventListener('error', () => {
      clearTimeout(timer);
      iframe.remove();
      const err = new Error(`IframeGuard: load error for "${src}"`);
      onError?.(err);
      reject(err);
    });

    container.appendChild(iframe);
  });
}
