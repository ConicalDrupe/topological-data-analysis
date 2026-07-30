// Mapper v2 cluster-detail extensions: field distributions + image gallery + lightbox.
// Wraps kmapper's own window.set_focus_node (defined in kmapper.js, loaded before this
// script) rather than replacing it, and appends new sections into the existing
// #tooltip_content_focus_node panel. Expects `clusterDetailsExtra` (per-node payload,
// keyed by the same node "name" kmapper itself uses) and `GALLERY_BATCH_SIZE` to already
// be declared in the same <script> tag before this code runs.

(function () {
  const galleryState = {};
  let lightboxImages = [];
  let lightboxIndex = 0;

  function ensureExtraSections() {
    const detailsEl = document.querySelector("#tooltip_content_focus_node .details");
    if (!detailsEl || detailsEl.querySelector(".cluster-details-extra")) return;

    const wrapper = document.createElement("div");
    wrapper.className = "cluster-details-extra";
    wrapper.innerHTML =
      "<hr><br/>" +
      "<h3>Field Distributions</h3>" +
      '<div class="field-distributions"></div>' +
      "<hr><br/>" +
      "<h3>Image Gallery</h3>" +
      '<div class="image-gallery"></div>';
    detailsEl.appendChild(wrapper);

    if (!document.querySelector(".cluster-lightbox-overlay")) {
      const lb = document.createElement("div");
      lb.className = "cluster-lightbox-overlay";
      lb.hidden = true;
      lb.innerHTML =
        '<span class="cluster-lightbox-close">&times;</span>' +
        '<span class="cluster-lightbox-nav cluster-lightbox-prev">&#8249;</span>' +
        '<span class="cluster-lightbox-nav cluster-lightbox-next">&#8250;</span>' +
        '<div class="cluster-lightbox-content">' +
        '<img src="" alt="">' +
        '<div class="cluster-lightbox-info"></div>' +
        "</div>";
      document.body.appendChild(lb);

      lb.querySelector(".cluster-lightbox-close").addEventListener("click", closeLightbox);
      lb.addEventListener("click", function (evt) {
        if (evt.target === lb) closeLightbox();
      });
      lb.querySelector(".cluster-lightbox-prev").addEventListener("click", function () {
        navigateLightbox(-1);
      });
      lb.querySelector(".cluster-lightbox-next").addEventListener("click", function () {
        navigateLightbox(1);
      });
      document.addEventListener("keydown", function (evt) {
        if (lb.hidden) return;
        if (evt.key === "Escape") closeLightbox();
        if (evt.key === "ArrowLeft") navigateLightbox(-1);
        if (evt.key === "ArrowRight") navigateLightbox(1);
      });
    }
  }

  function openLightbox(images, index) {
    lightboxImages = images;
    lightboxIndex = index;
    renderLightbox();
    document.querySelector(".cluster-lightbox-overlay").hidden = false;
  }

  function closeLightbox() {
    const overlay = document.querySelector(".cluster-lightbox-overlay");
    if (overlay) overlay.hidden = true;
  }

  function navigateLightbox(delta) {
    if (!lightboxImages.length) return;
    lightboxIndex = (lightboxIndex + delta + lightboxImages.length) % lightboxImages.length;
    renderLightbox();
  }

  function renderLightbox() {
    const img = lightboxImages[lightboxIndex];
    const overlay = document.querySelector(".cluster-lightbox-overlay");
    overlay.querySelector("img").src = img.path;
    overlay.querySelector(".cluster-lightbox-info").textContent =
      img.tooltip + " (" + (lightboxIndex + 1) + " / " + lightboxImages.length + ")";
  }

  function renderDistributions(container, distributions) {
    container.innerHTML = "";
    Object.keys(distributions).forEach(function (field) {
      const dist = distributions[field];
      const block = document.createElement("div");
      block.className = "field-distribution";

      const nameEl = document.createElement("div");
      nameEl.className = "field-name";
      nameEl.textContent = field;
      block.appendChild(nameEl);

      if (dist.type === "categorical") {
        const counts = Object.values(dist.counts).concat([1]);
        const maxCount = Math.max.apply(null, counts);
        Object.keys(dist.counts).forEach(function (value) {
          const count = dist.counts[value];
          const pct = dist.proportions[value] * 100;
          const row = document.createElement("div");
          row.className = "dist-bar-row";
          row.innerHTML =
            '<div class="dist-bar-label" title="' + value + '">' + value + "</div>" +
            '<div class="dist-bar-track"><div class="dist-bar-fill" style="width:' +
            (count / maxCount) * 100 +
            '%"></div></div>' +
            '<div class="dist-bar-value">' + count + " (" + pct.toFixed(0) + "%)</div>";
          block.appendChild(row);
        });
        if (dist.missing) {
          const missingEl = document.createElement("div");
          missingEl.className = "field-stats";
          missingEl.textContent = dist.missing + " missing";
          block.appendChild(missingEl);
        }
      } else {
        const statsEl = document.createElement("div");
        statsEl.className = "field-stats";
        if (dist.mean !== null) {
          statsEl.textContent =
            "mean " + dist.mean.toFixed(2) +
            ", median " + dist.median.toFixed(2) +
            ", std " + dist.std.toFixed(2) +
            ", range [" + dist.min.toFixed(2) + ", " + dist.max.toFixed(2) + "]";
        } else {
          statsEl.textContent = "no data";
        }
        block.appendChild(statsEl);

        const binCounts = dist.histogram.map(function (b) {
          return b.count;
        });
        const maxCount = Math.max.apply(null, binCounts.concat([1]));
        dist.histogram.forEach(function (bin) {
          const row = document.createElement("div");
          row.className = "dist-bar-row";
          const label = bin.bin_start.toFixed(1) + "-" + bin.bin_end.toFixed(1);
          row.innerHTML =
            '<div class="dist-bar-label" title="' + label + '">' + label + "</div>" +
            '<div class="dist-bar-track"><div class="dist-bar-fill" style="width:' +
            (bin.count / maxCount) * 100 +
            '%"></div></div>' +
            '<div class="dist-bar-value">' + bin.count + "</div>";
          block.appendChild(row);
        });
      }

      container.appendChild(block);
    });
  }

  function renderGalleryBatch(container, images, nodeId) {
    const state = galleryState[nodeId];
    const grid = container.querySelector(".image-gallery-grid");
    const start = state.rendered;
    const end = Math.min(images.length, start + GALLERY_BATCH_SIZE);

    for (let i = start; i < end; i++) {
      const img = images[i];
      const el = document.createElement("img");
      el.className = "image-gallery-thumb";
      el.src = img.path;
      el.title = img.tooltip;
      el.loading = "lazy";
      el.addEventListener(
        "click",
        (function (idx) {
          return function () {
            openLightbox(images, idx);
          };
        })(i)
      );
      grid.appendChild(el);
    }
    state.rendered = end;

    const existingBtn = container.querySelector(".image-gallery-loadmore");
    if (existingBtn) existingBtn.remove();
    if (state.rendered < images.length) {
      const btn = document.createElement("button");
      btn.className = "image-gallery-loadmore";
      btn.textContent =
        "Load " + Math.min(GALLERY_BATCH_SIZE, images.length - state.rendered) +
        " more (" + (images.length - state.rendered) + " remaining)";
      btn.addEventListener("click", function () {
        renderGalleryBatch(container, images, nodeId);
      });
      container.appendChild(btn);
    }
  }

  function renderGallery(container, images, nodeId) {
    if (!images.length) {
      container.innerHTML = '<div class="image-gallery-empty">No images available for this cluster.</div>';
      return;
    }
    container.innerHTML = '<div class="image-gallery-grid"></div>';
    galleryState[nodeId] = { rendered: 0 };
    renderGalleryBatch(container, images, nodeId);
  }

  function renderClusterExtras(nodeId, data) {
    ensureExtraSections();
    const distContainer = document.querySelector("#tooltip_content_focus_node .field-distributions");
    const galleryContainer = document.querySelector("#tooltip_content_focus_node .image-gallery");
    if (distContainer) renderDistributions(distContainer, data.distributions);
    if (galleryContainer) renderGallery(galleryContainer, data.images, nodeId);
  }

  const originalSetFocusNode = window.set_focus_node;
  window.set_focus_node = function (d) {
    originalSetFocusNode(d);
    if (d && typeof clusterDetailsExtra !== "undefined" && clusterDetailsExtra[d.name]) {
      renderClusterExtras(d.name, clusterDetailsExtra[d.name]);
    }
  };
})();
