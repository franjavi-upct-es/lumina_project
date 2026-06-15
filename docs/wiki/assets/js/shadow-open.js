/*
 * Material for MkDocs renders each Mermaid diagram into a CLOSED shadow root
 * attached to a `div.mermaid` host:
 *
 *   let r = createElement("div.mermaid");
 *   let s = r.attachShadow({ mode: "closed" });
 *   s.innerHTML = svg;
 *
 * A closed shadow root is unreachable from scripts (`host.shadowRoot` is null),
 * so mermaid-zoom.js cannot find the <svg> to attach pan/zoom to.
 *
 * Patch Element.prototype.attachShadow so these roots are created `open`
 * instead. Material calls attachShadow asynchronously (after the diagram
 * renders), and this script runs synchronously at load — well before — so the
 * patch is in place in time. Open vs closed only affects script access, not
 * styling or rendering, so Material is unaffected.
 */
(function () {
  "use strict";
  var native = Element.prototype.attachShadow;
  if (typeof native !== "function") {
    return;
  }
  Element.prototype.attachShadow = function (init) {
    if (init && init.mode === "closed") {
      init = Object.assign({}, init, { mode: "open" });
    }
    return native.call(this, init);
  };
})();
