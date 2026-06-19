/*
 * Replicates the claude.ai Mermaid card: each diagram becomes a framed card
 * with a hover-revealed expand control in the top-right corner.
 *
 * Inline cards are intentionally static — no wheel/trackpad zoom, no pan — and
 * clicking anywhere on the diagram opens the augmentation window (same as the
 * expand button). That window shows a fresh copy with button-driven zoom and
 * drag-to-pan (still no wheel zoom), done by mutating the <svg>'s `viewBox`
 * directly (not a library).
 *
 * Material for MkDocs renders each diagram's <svg> inside a shadow root on a
 * `div.mermaid` host (see shadow-open.js, which forces those roots open). Shadow
 * mutations don't bubble to a document-level observer and Material renders
 * asynchronously, so we poll for diagrams and rescan on instant navigation.
 */
(function () {
    "use strict";

    var POLL_MS = 300;
    var POLL_WINDOW_MS = 20000;
    var MIN_SCALE = 1;
    var MAX_SCALE = 25;
    var STEP = 1.2; // button zoom factor
    var FILL_VH = 80; // inline target height; keep in sync with .mz-viewport max-height

    var SVG_NS = "http://www.w3.org/2000/svg";

    // Icon paths lifted from the reference component (plus / minus / expand).
    var ICONS = {
        zoomIn: "M11.700 3.076 C 11.506 3.165,11.384 3.292,11.302 3.489 C 11.248 3.619,11.240 4.098,11.240 7.438 L 11.240 11.238 7.390 11.249 C 3.622 11.260,3.537 11.262,3.404 11.340 C 3.330 11.384,3.213 11.489,3.144 11.574 C 3.032 11.712,3.020 11.755,3.020 12.000 C 3.020 12.244,3.033 12.289,3.142 12.426 C 3.210 12.511,3.344 12.620,3.441 12.670 L 3.617 12.760 7.428 12.760 L 11.240 12.760 11.240 16.572 L 11.240 20.383 11.330 20.559 C 11.380 20.656,11.489 20.790,11.574 20.858 C 11.711 20.967,11.756 20.980,12.000 20.980 C 12.244 20.980,12.289 20.967,12.426 20.858 C 12.511 20.790,12.620 20.656,12.670 20.559 L 12.760 20.383 12.760 16.572 L 12.760 12.760 16.572 12.760 L 20.383 12.760 20.559 12.670 C 20.656 12.621,20.790 12.511,20.858 12.426 C 20.967 12.289,20.980 12.244,20.980 12.000 C 20.980 11.755,20.968 11.712,20.856 11.574 C 20.787 11.489,20.670 11.384,20.596 11.340 C 20.463 11.262,20.378 11.260,16.610 11.249 L 12.760 11.238 12.759 7.429 C 12.758 3.811,12.755 3.612,12.684 3.460 C 12.503 3.068,12.077 2.902,11.700 3.076",
        zoomOut:
            "M6.571 11.278 C 6.395 11.321,6.266 11.411,6.133 11.586 C 6.035 11.714,6.020 11.769,6.020 12.003 C 6.020 12.243,6.033 12.289,6.142 12.426 C 6.210 12.511,6.344 12.620,6.441 12.670 L 6.617 12.760 12.000 12.760 L 17.383 12.760 17.559 12.670 C 17.656 12.620,17.790 12.511,17.858 12.426 C 17.967 12.289,17.980 12.244,17.980 12.000 C 17.980 11.755,17.968 11.712,17.856 11.574 C 17.787 11.489,17.670 11.384,17.596 11.340 C 17.462 11.261,17.382 11.260,12.080 11.254 C 9.121 11.250,6.642 11.261,6.571 11.278",
        expand: "M5.270 3.041 C 4.702 3.138,4.154 3.442,3.728 3.898 C 3.450 4.195,3.247 4.539,3.114 4.940 C 3.021 5.219,3.021 5.225,3.008 7.760 C 3.001 9.157,3.007 10.360,3.020 10.432 C 3.050 10.594,3.293 10.873,3.469 10.946 C 3.748 11.063,4.126 10.967,4.328 10.727 C 4.386 10.658,4.452 10.511,4.476 10.399 C 4.505 10.266,4.520 9.383,4.520 7.828 L 4.521 5.460 4.623 5.240 C 4.758 4.948,4.929 4.775,5.220 4.635 L 5.460 4.520 12.500 4.520 L 19.540 4.521 19.760 4.623 C 20.052 4.758,20.225 4.929,20.365 5.220 L 20.480 5.460 20.480 9.500 L 20.480 13.540 20.366 13.782 C 20.231 14.066,19.962 14.321,19.688 14.425 C 19.517 14.490,19.271 14.499,16.977 14.518 L 14.454 14.540 14.302 14.655 C 14.039 14.857,13.931 15.237,14.054 15.531 C 14.127 15.707,14.406 15.950,14.568 15.980 C 14.640 15.993,15.843 15.999,17.240 15.992 C 19.775 15.979,19.781 15.979,20.060 15.886 C 20.511 15.736,20.831 15.536,21.184 15.184 C 21.545 14.822,21.744 14.498,21.887 14.040 L 21.980 13.740 21.980 9.500 L 21.980 5.260 21.887 4.960 C 21.625 4.118,20.939 3.413,20.109 3.131 L 19.780 3.020 12.620 3.014 C 8.682 3.011,5.375 3.023,5.270 3.041 M4.270 13.040 C 3.699 13.139,3.154 13.443,2.728 13.898 C 2.440 14.206,2.241 14.550,2.113 14.960 C 2.020 15.259,2.020 15.266,2.020 17.500 C 2.020 19.734,2.020 19.741,2.113 20.040 C 2.256 20.498,2.455 20.822,2.816 21.184 C 3.178 21.545,3.502 21.744,3.960 21.887 L 4.260 21.980 7.000 21.980 L 9.740 21.980 10.040 21.887 C 10.498 21.744,10.822 21.545,11.184 21.184 C 11.545 20.822,11.744 20.498,11.887 20.040 C 11.980 19.741,11.980 19.734,11.980 17.500 C 11.980 15.266,11.980 15.259,11.887 14.960 C 11.625 14.118,10.939 13.412,10.109 13.132 L 9.780 13.021 7.120 13.014 C 5.657 13.011,4.375 13.022,4.270 13.040 M9.780 14.635 C 10.071 14.775,10.242 14.948,10.377 15.240 L 10.479 15.460 10.480 17.500 L 10.480 19.540 10.366 19.782 C 10.226 20.076,9.954 20.327,9.667 20.428 C 9.476 20.494,9.268 20.500,7.000 20.500 C 4.732 20.500,4.524 20.494,4.333 20.428 C 4.046 20.327,3.774 20.076,3.634 19.782 L 3.520 19.540 3.520 17.500 L 3.521 15.460 3.623 15.240 C 3.758 14.948,3.929 14.775,4.220 14.635 L 4.460 14.520 7.000 14.520 L 9.540 14.520 9.780 14.635",
        close: "M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z",
    };

    function iconButton(label, pathD) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "mz-btn";
        btn.setAttribute("aria-label", label);
        var svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", "0 0 25 25");
        svg.setAttribute("aria-hidden", "true");
        var path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("d", pathD);
        path.setAttribute("fill", "currentColor");
        svg.appendChild(path);
        btn.appendChild(svg);
        return btn;
    }

    // Parse "minX minY width height" into an object.
    function parseViewBox(svg) {
        var vb = (svg.getAttribute("viewBox") || "")
            .split(/[\s,]+/)
            .map(Number);
        if (vb.length !== 4 || vb.some(isNaN)) {
            var b = svg.getBBox
                ? svg.getBBox()
                : { x: 100, y: 100, width: 100, height: 100 };
            return { x: b.x, y: b.y, w: b.width, h: b.height };
        }
        return { x: vb[0], y: vb[1], w: vb[2], h: vb[3] };
    }

    // `opts.wheel` enables mouse/trackpad wheel zoom; `opts.pan` enables
    // drag-to-pan, double-click-to-reset and the grab cursor. Inline cards pass
    // neither (static, click-to-expand); the augmentation window enables pan.
    function setup(svg, opts) {
        if (svg.dataset.mzInit) {
            return;
        }
        svg.dataset.mzInit = "1";

        opts = opts || {};
        var allowWheel = !!opts.wheel;
        var allowPan = !!opts.pan;

        var base = parseViewBox(svg);
        var cur = { x: base.x, y: base.y, w: base.w, h: base.h };

        // Remember the fit viewBox so the augmentation window can reset to it.
        svg.dataset.mzBase = base.x + " " + base.y + " " + base.w + " " + base.h;

        svg.style.maxWidth = "100%";
        svg.style.maxHeight = "100%";
        svg.style.width = "100%";
        svg.style.height = "auto";
        svg.style.display = "block";
        svg.style.cursor = allowPan ? "grab" : "pointer";
        if (allowPan) {
            svg.style.touchAction = "none";
        }
        svg.style.userSelect = "none";

        function apply() {
            svg.setAttribute(
                "viewBox",
                cur.x + " " + cur.y + " " + cur.w + " " + cur.h,
            );
        }

        // Zoom by `factor` (>1 zooms out) keeping user-space point (cx,cy) fixed.
        function zoom(factor, cx, cy) {
            var nextW = cur.w * factor;
            var scale = base.w / nextW;
            if (scale < MIN_SCALE || scale > MAX_SCALE) {
                return;
            }
            cur.x = cx - (cx - cur.x) * factor;
            cur.y = cy - (cy - cur.y) * factor;
            cur.w = nextW;
            cur.h = cur.h * factor;
            apply();
        }

        function center() {
            return { cx: cur.x + cur.w / 2, cy: cur.y + cur.h / 2 };
        }

        function reset() {
            cur = { x: base.x, y: base.y, w: base.w, h: base.h };
            apply();
        }

        // Map a client point to current user-space coordinates.
        function toUser(clientX, clientY) {
            var r = svg.getBoundingClientRect();
            return {
                x: cur.x + ((clientX - r.left) / r.width) * cur.w,
                y: cur.y + ((clientY - r.top) / r.height) * cur.h,
            };
        }

        // Mouse/trackpad wheel zoom — only when explicitly enabled.
        if (allowWheel) {
            svg.addEventListener(
                "wheel",
                function (e) {
                    e.preventDefault();
                    var p = toUser(e.clientX, e.clientY);
                    zoom(e.deltaY > 0 ? 1.1 : 1 / 1.1, p.x, p.y);
                },
                { passive: false },
            );
        }

        // Drag-to-pan and double-click-to-reset — only when enabled.
        if (allowPan) {
            var dragging = false;
            var last = null;
            svg.addEventListener("pointerdown", function (e) {
                if (e.button !== 0) {
                    return;
                }
                dragging = true;
                last = { x: e.clientX, y: e.clientY };
                svg.style.cursor = "grabbing";
                svg.setPointerCapture(e.pointerId);
            });
            svg.addEventListener("pointermove", function (e) {
                if (!dragging) {
                    return;
                }
                var r = svg.getBoundingClientRect();
                cur.x -= ((e.clientX - last.x) / r.width) * cur.w;
                cur.y -= ((e.clientY - last.y) / r.height) * cur.h;
                last = { x: e.clientX, y: e.clientY };
                apply();
            });
            var endDrag = function () {
                dragging = false;
                svg.style.cursor = "grab";
            };
            svg.addEventListener("pointerup", endDrag);
            svg.addEventListener("pointercancel", endDrag);
            svg.addEventListener("dblclick", reset);
        }

        return {
            zoomIn: function () {
                var c = center();
                zoom(1 / STEP, c.cx, c.cy);
            },
            zoomOut: function () {
                var c = center();
                zoom(STEP, c.cx, c.cy);
            },
            reset: reset,
        };
    }

    // The augmentation window: a large centered modal showing a fresh copy of
    // the diagram with its own pan/zoom and a close control.
    function openModal(svg) {
        if (document.querySelector(".mz-modal")) {
            return; // one augmentation window at a time
        }
        var overlay = document.createElement("div");
        overlay.className = "mz-modal";

        var panel = document.createElement("div");
        panel.className = "mz-modal-panel";

        var viewport = document.createElement("div");
        viewport.className = "mz-modal-viewport";

        // Clone the diagram so the inline card keeps its own state. Reset the
        // clone to the fit viewBox and clear the init flag so setup() re-wires
        // pan/zoom on it.
        var clone = svg.cloneNode(true);
        delete clone.dataset.mzInit;
        if (svg.dataset.mzBase) {
            clone.setAttribute("viewBox", svg.dataset.mzBase);
        }
        viewport.appendChild(clone);

        // Button-driven zoom + drag-to-pan inside the window, but still no
        // wheel/trackpad zoom.
        var api = setup(clone, { pan: true });
        clone.style.height = "100%";

        var controls = document.createElement("div");
        controls.className = "mz-controls";
        var zin = iconButton("Zoom in", ICONS.zoomIn);
        var zout = iconButton("Zoom out", ICONS.zoomOut);
        var close = iconButton("Close", ICONS.close);
        zin.addEventListener("click", api.zoomIn);
        zout.addEventListener("click", api.zoomOut);
        controls.appendChild(zin);
        controls.appendChild(zout);
        controls.appendChild(close);

        panel.appendChild(viewport);
        panel.appendChild(controls);
        overlay.appendChild(panel);

        function destroy() {
            overlay.remove();
            document.removeEventListener("keydown", onKey);
            document.body.classList.remove("mz-modal-open");
        }
        function onKey(e) {
            if (e.key === "Escape") {
                destroy();
            }
        }
        close.addEventListener("click", destroy);
        overlay.addEventListener("click", function (e) {
            if (e.target === overlay) {
                destroy(); // click the backdrop to dismiss
            }
        });
        document.addEventListener("keydown", onKey);
        document.body.classList.add("mz-modal-open");
        document.body.appendChild(overlay);
    }

    // Scale the inline diagram to fill its card as large as possible while
    // preserving aspect ratio ("contain", but scaled up). A Mermaid <svg> has a
    // small intrinsic size, so width/height:auto leaves it tiny — we must set an
    // explicit dimension to force scale-up, picking whichever one is the binding
    // constraint for the card's box. This makes every diagram (LR or TB) large.
    function sizeToFill(host, svg, viewport) {
        var vb = (svg.dataset.mzBase || "").split(/\s+/).map(Number);
        if (vb.length !== 4 || !vb[2] || !vb[3]) {
            return;
        }
        var diagramAspect = vb[2] / vb[3];
        var boxW = viewport.clientWidth;
        var boxH = window.innerHeight * (FILL_VH / 100);

        // The .mermaid host is a flex item that otherwise shrink-wraps the
        // diagram's small natural size — so `svg { width: 100% }` resolves
        // against a tiny host and stays tiny. Stretch the host to fill the
        // viewport first; only then does the svg have room to scale up into.
        host.style.display = "block";
        host.style.width = "100%";

        svg.style.maxWidth = "100%";
        svg.style.maxHeight = FILL_VH + "vh";
        svg.style.margin = "0 auto"; // center when narrower than the host
        if (!boxW || diagramAspect >= boxW / boxH) {
            // Wider than the box → bounded by width: fill it.
            svg.style.width = "100%";
            svg.style.height = "auto";
        } else {
            // Taller than the box → bounded by height: fill it.
            svg.style.width = "auto";
            svg.style.height = FILL_VH + "vh";
        }
    }

    function wrap(host, svg) {
        if (host.closest(".mz-card")) {
            return; // already wrapped
        }

        var card = document.createElement("div");
        card.className = "mz-card";

        var viewport = document.createElement("div");
        viewport.className = "mz-viewport";

        var controls = document.createElement("div");
        controls.className = "mz-controls";

        // Inline cards are static, so they expose only the expand control.
        var expand = iconButton("Expand", ICONS.expand);
        expand.addEventListener("click", function (e) {
            e.stopPropagation();
            openModal(svg);
        });
        controls.appendChild(expand);

        host.replaceWith(card);
        viewport.appendChild(host);
        card.appendChild(viewport);
        card.appendChild(controls);

        // Clicking anywhere on the diagram frame opens the augmentation window,
        // mirroring the expand button.
        viewport.addEventListener("click", function () {
            openModal(svg);
        });

        // Card is now in the document, so the viewport has a real width to
        // measure against — size the diagram to fill it.
        sizeToFill(host, svg, viewport);
    }

    function scan() {
        var hosts = document.querySelectorAll(".mermaid");
        var pending = 0;
        hosts.forEach(function (host) {
            var root = host.shadowRoot || host;
            var svg = root.querySelector("svg");
            if (svg) {
                // Inline diagrams are static (no wheel/pan); click to expand.
                var api = setup(svg, {});
                if (api) {
                    wrap(host, svg);
                }
            } else {
                pending++;
            }
        });
        return hosts.length > 0 && pending === 0;
    }

    var timer = null;
    function startPolling() {
        if (timer) {
            clearInterval(timer);
        }
        var deadline = Date.now() + POLL_WINDOW_MS;
        timer = setInterval(function () {
            if (scan() || Date.now() > deadline) {
                clearInterval(timer);
                timer = null;
            }
        }, POLL_MS);
    }

    if (window.document$ && typeof window.document$.subscribe === "function") {
        window.document$.subscribe(startPolling);
    } else {
        window.addEventListener("load", startPolling);
    }
})();
