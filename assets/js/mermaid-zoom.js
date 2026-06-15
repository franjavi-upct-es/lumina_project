/*
 * Adds interactive pan & zoom controls (+ / - / reset) to every Mermaid
 * diagram so wide left-to-right flowcharts can be inspected instead of being
 * squished to fit the page width.
 *
 * Material renders Mermaid SVGs asynchronously and re-renders on instant
 * navigation, so we (a) rescan on each `document$` emission and (b) keep a
 * MutationObserver running to catch SVGs that mount after the scan.
 */
(function () {
  "use strict";

  var DIAGRAM_HEIGHT = "480px";

  function enhance(svg) {
    if (svg.dataset.panzoom || typeof window.svgPanZoom === "undefined") {
      return;
    }
    svg.dataset.panzoom = "1";

    // Mermaid pins an inline max-width and intrinsic height; give the SVG a
    // stable viewport box so pan/zoom has room to work.
    svg.style.maxWidth = "100%";
    svg.style.width = "100%";
    svg.style.height = DIAGRAM_HEIGHT;

    window.svgPanZoom(svg, {
      zoomEnabled: true,
      controlIconsEnabled: true,
      fit: true,
      center: true,
      minZoom: 0.4,
      maxZoom: 20,
      zoomScaleSensitivity: 0.3,
    });
  }

  function scan() {
    document.querySelectorAll(".mermaid > svg").forEach(enhance);
  }

  function init() {
    scan();
    new MutationObserver(scan).observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    // Material instant navigation: rescan after each page swap.
    window.document$.subscribe(function () {
      // Mermaid renders a tick after the DOM swap; the observer covers the
      // rest, this just speeds up the common case.
      setTimeout(scan, 300);
    });
    init();
  } else {
    window.addEventListener("load", init);
  }
})();
