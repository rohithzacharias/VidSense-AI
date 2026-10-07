/**
 * VidSense AI — Frontend Application Logic
 * Integrates Video Player, MongoDB Event Timeline, and Groq LLM QA.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const videoSelect = document.getElementById("videoSelect");
  const videoPlayer = document.getElementById("mainVideoPlayer");
  const videoFileInput = document.getElementById("videoFileInput");
  const videoStatusBadge = document.getElementById("videoStatusBadge");
  const currentVideoName = document.getElementById("currentVideoName");
  
  const currentTimeDisplay = document.getElementById("currentTimeDisplay");
  const durationDisplay = document.getElementById("durationDisplay");
  const currentEventTitle = document.getElementById("currentEventTitle");
  const currentEventOverlay = document.getElementById("currentEventOverlay");
  const eventOverlayText = document.getElementById("eventOverlayText");
  const eventCountBadge = document.getElementById("eventCountBadge");
  const timelineList = document.getElementById("timelineList");

  const processVideoBtn = document.getElementById("processVideoBtn");
  const questionInput = document.getElementById("questionInput");
  const askButton = document.getElementById("askButton");
  const useCurrentTimeCheckbox = document.getElementById("useCurrentTimeCheckbox");
  const activeFilterTime = document.getElementById("activeFilterTime");
  const pillButtons = document.querySelectorAll(".pill-btn");

  const answerPlaceholder = document.getElementById("answerPlaceholder");
  const answerCard = document.getElementById("answerCard");
  const answerText = document.getElementById("answerText");
  const answerTimeRange = document.getElementById("answerTimeRange");
  const keyObjectsContainer = document.getElementById("keyObjectsContainer");
  const objectsTagsList = document.getElementById("objectsTagsList");
  const evidenceFramesGrid = document.getElementById("evidenceFramesGrid");
  const jumpToEventBtn = document.getElementById("jumpToEventBtn");

  // State
  let currentVideoId = "";
  let loadedEvents = [];
  let currentAnswerStart = 0.0;
  let isProcessing = false;
  let pollInterval = null;

  // Format seconds to mm:ss.d
  function formatTime(seconds) {
    if (isNaN(seconds) || seconds === null) return "00:00.0";
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(1);
    const paddedSecs = secs < 10 ? `0${secs}` : secs;
    return `${mins < 10 ? '0' : ''}${mins}:${paddedSecs}`;
  }

  // Format range
  function formatRange(start, end) {
    const s = Math.floor(start);
    const e = Math.floor(end);
    const sMin = Math.floor(s / 60);
    const sSec = s % 60;
    const eMin = Math.floor(e / 60);
    const eSec = e % 60;
    const pad = (n) => (n < 10 ? `0${n}` : n);
    return `${pad(sMin)}:${pad(sSec)} — ${pad(eMin)}:${pad(eSec)}`;
  }

  // Fetch Videos List
  async function loadVideos() {
    try {
      const res = await fetch("/api/videos");
      if (!res.ok) throw new Error("Failed to load videos");
      const data = await res.json();
      const videos = data.videos || [];

      videoSelect.innerHTML = "";
      if (videos.length === 0) {
        videoSelect.innerHTML = `<option value="">No videos found</option>`;
        return;
      }

      videos.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v.video_id;
        opt.textContent = `${v.filename} (${v.event_count} events)`;
        opt.dataset.url = v.url;
        opt.dataset.duration = v.total_duration;
        videoSelect.appendChild(opt);
      });

      // Default to 'skating' if present, else first
      const skatingOption = Array.from(videoSelect.options).find(o => o.value === "skating");
      if (skatingOption) {
        videoSelect.value = "skating";
      } else {
        videoSelect.value = videos[0].video_id;
      }

      onVideoSelected(videoSelect.value);
    } catch (err) {
      console.error(err);
      videoSelect.innerHTML = `<option value="">Error loading videos</option>`;
    }
  }

  // Handle Video Selection
  async function onVideoSelected(videoId) {
    if (!videoId) return;
    currentVideoId = videoId;
    currentVideoName.textContent = `${videoId}.mp4`;

    const selectedOption = videoSelect.selectedOptions[0];
    const videoUrl = selectedOption ? selectedOption.dataset.url : `/data/videos/uploads/${videoId}.mp4`;

    // Set video src
    videoPlayer.src = videoUrl;
    videoPlayer.load();

    // Fetch Events from MongoDB
    await loadEvents(videoId);
  }

  // Load Events from MongoDB via API
  async function loadEvents(videoId) {
    timelineList.innerHTML = `
      <div class="timeline-loading">
        <div class="spinner"></div>
        <span>Querying events from MongoDB...</span>
      </div>
    `;

    try {
      const res = await fetch(`/api/events/${videoId}`);
      if (!res.ok) throw new Error("Failed to fetch events");
      const data = await res.json();
      loadedEvents = data.events || [];

      eventCountBadge.textContent = `${loadedEvents.length} Events`;
      renderTimeline(loadedEvents);
    } catch (err) {
      console.error(err);
      timelineList.innerHTML = `
        <div class="timeline-loading">
          <span style="color:#f87171;">⚠️ Failed to load events from MongoDB.</span>
        </div>
      `;
    }
  }

  // Trigger Video Pipeline Execution
  async function triggerProcessVideo() {
    if (!currentVideoId || isProcessing) return;

    isProcessing = true;
    processVideoBtn.disabled = true;
    processVideoBtn.innerHTML = `<span class="spinner" style="width:14px;height:14px;"></span> Processing...`;
    videoStatusBadge.className = "badge status-badge";
    videoStatusBadge.innerHTML = `<span class="spinner" style="width:10px;height:10px;"></span> Running YOLO & Qwen...`;

    timelineList.innerHTML = `
      <div class="timeline-loading" style="flex-direction:column;gap:16px;">
        <div class="spinner" style="width:36px;height:36px;"></div>
        <div style="text-align:center;">
          <h4 style="color:#f8fafc;font-size:0.95rem;margin-bottom:4px;">Extracting Video Understanding...</h4>
          <p style="color:var(--text-muted);font-size:0.8rem;">Sampling 10 FPS -> YOLO Detection -> BoT-SORT Tracking -> Qwen2.5-VL</p>
        </div>
      </div>
    `;

    try {
      const res = await fetch(`/api/process/${currentVideoId}`, { method: "POST" });
      if (!res.ok) throw new Error("Failed to start processing");

      // Start polling status
      pollInterval = setInterval(async () => {
        try {
          const sRes = await fetch(`/api/status/${currentVideoId}`);
          const sData = await sRes.json();
          if (sData.status === "completed") {
            clearInterval(pollInterval);
            isProcessing = false;
            processVideoBtn.disabled = false;
            processVideoBtn.innerHTML = `<span>⚡</span> Process Video`;
            videoStatusBadge.innerHTML = `<span class="status-dot"></span> Ready (${sData.event_count} events)`;
            await loadEvents(currentVideoId);
            await loadVideos();
          } else if (sData.status === "failed") {
            clearInterval(pollInterval);
            isProcessing = false;
            processVideoBtn.disabled = false;
            processVideoBtn.innerHTML = `<span>⚡</span> Retry Process`;
            videoStatusBadge.innerHTML = `<span style="color:#ef4444;">Failed</span>`;
            alert(`Pipeline failed: ${sData.message}`);
          }
        } catch (e) {
          console.error(e);
        }
      }, 3000);

    } catch (err) {
      console.error(err);
      isProcessing = false;
      processVideoBtn.disabled = false;
      processVideoBtn.innerHTML = `<span>⚡</span> Process Video`;
      alert("Failed to start processing pipeline.");
    }
  }

  // Render Interactive Timeline
  function renderTimeline(events) {
    if (events.length === 0) {
      videoStatusBadge.innerHTML = `<span class="status-dot" style="background:#f59e0b;box-shadow:0 0 8px #f59e0b;"></span> Unprocessed`;
      timelineList.innerHTML = `
        <div class="timeline-loading" style="flex-direction:column;gap:14px;padding:32px 16px;">
          <span style="font-size:1.8rem;">⚡</span>
          <div style="text-align:center;">
            <p style="color:#f8fafc;font-size:0.95rem;font-weight:600;margin-bottom:4px;">No events extracted for this video yet.</p>
            <p style="color:var(--text-muted);font-size:0.8rem;margin-bottom:14px;">Run the local pipeline to detect objects, track movements, and reason with Qwen2.5-VL.</p>
            <button id="inlineProcessBtn" class="btn btn-primary" style="padding:8px 20px;font-size:0.85rem;">
              <span>⚡</span> Process '${currentVideoId}.mp4' Now
            </button>
          </div>
        </div>
      `;
      const inlineBtn = document.getElementById("inlineProcessBtn");
      if (inlineBtn) {
        inlineBtn.addEventListener("click", triggerProcessVideo);
      }
      return;
    }

    videoStatusBadge.innerHTML = `<span class="status-dot"></span> Ready (${events.length} events)`;
    timelineList.innerHTML = "";
    events.forEach((evt, idx) => {
      const item = document.createElement("div");
      item.className = "timeline-item";
      item.dataset.index = idx;
      item.dataset.start = evt.start_time;
      item.dataset.end = evt.end_time;

      const timeBadge = formatRange(evt.start_time, evt.end_time);
      const tracksText = evt.track_ids && evt.track_ids.length > 0 
        ? evt.track_ids.slice(0, 4).map(t => `#${t}`).join(", ") 
        : "None";

      item.innerHTML = `
        <div class="timeline-time-badge">${timeBadge}</div>
        <div class="timeline-item-body">
          <p class="timeline-item-desc">${evt.description || "Activity observed in frame sequence."}</p>
          <div class="timeline-item-meta">
            <span class="timeline-tag">${evt.event_type.toUpperCase()}</span>
            <span>Tracks: ${tracksText}</span>
            <span>Conf: ${(evt.confidence * 100).toFixed(0)}%</span>
          </div>
        </div>
      `;

      item.addEventListener("click", () => {
        // Jump video to event start
        videoPlayer.currentTime = evt.start_time;
        videoPlayer.play();
        highlightActiveTimeline(idx);
      });

      timelineList.appendChild(item);
    });
  }

  // Highlight Active Timeline Item
  function highlightActiveTimeline(activeIndex) {
    const items = timelineList.querySelectorAll(".timeline-item");
    items.forEach((it, idx) => {
      if (idx === activeIndex) {
        it.classList.add("active");
        it.scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else {
        it.classList.remove("active");
      }
    });
  }

  // Live Playback Updates
  videoPlayer.addEventListener("timeupdate", () => {
    const cur = videoPlayer.currentTime;
    const dur = videoPlayer.duration || 10.0;

    currentTimeDisplay.textContent = formatTime(cur);
    durationDisplay.textContent = formatTime(dur);
    activeFilterTime.textContent = formatTime(cur);

    // Find current event
    const currentEventIdx = loadedEvents.findIndex(
      (e) => cur >= e.start_time && cur < e.end_time
    );

    if (currentEventIdx !== -1) {
      const evt = loadedEvents[currentEventIdx];
      currentEventTitle.textContent = `${evt.event_type.toUpperCase()} (${formatRange(evt.start_time, evt.end_time)})`;
      eventOverlayText.textContent = evt.description ? evt.description.slice(0, 60) + "..." : "Active Event";
      currentEventOverlay.style.display = "flex";
      highlightActiveTimeline(currentEventIdx);
    } else {
      currentEventOverlay.style.display = "none";
      currentEventTitle.textContent = "Between events";
    }
  });

  // Video Selector Change
  videoSelect.addEventListener("change", (e) => {
    onVideoSelected(e.target.value);
  });

  // Handle Video Upload
  videoFileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    videoStatusBadge.innerHTML = `<span class="spinner" style="width:12px;height:12px;"></span> Uploading...`;
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload?auto_process=false", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      videoStatusBadge.innerHTML = `<span class="status-dot"></span> Ready`;
      alert(`Video uploaded: ${data.filename}! You can select it from the dropdown.`);
      await loadVideos();
    } catch (err) {
      console.error(err);
      videoStatusBadge.innerHTML = `<span style="color:#ef4444;">Upload Failed</span>`;
      alert("Failed to upload video.");
    }
  });

  // Ask Question Handler (Groq Reasoning)
  async function handleAskQuestion() {
    const query = questionInput.value.trim();
    if (!query) {
      questionInput.focus();
      return;
    }

    if (!currentVideoId) {
      alert("Please select a video first.");
      return;
    }

    if (loadedEvents.length === 0) {
      if (confirm(`Video '${currentVideoId}' has not been processed yet!\n\nWould you like to process it now with YOLO & Qwen2.5-VL?`)) {
        triggerProcessVideo();
      }
      return;
    }

    // Set UI loading state
    askButton.disabled = true;
    askButton.innerHTML = `<div class="spinner" style="width:14px;height:14px;"></div> Reasoning...`;

    const payload = {
      video_id: currentVideoId,
      query: query,
    };

    if (useCurrentTimeCheckbox.checked) {
      payload.current_time = videoPlayer.currentTime;
      // Focus within +/- 5 seconds of playhead
      payload.time_range_start = Math.max(0, videoPlayer.currentTime - 5.0);
      payload.time_range_end = videoPlayer.currentTime + 5.0;
    }

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Error querying Groq");
      }

      const data = await res.json();
      displayAnswer(data);
    } catch (err) {
      console.error(err);
      alert(`Error: ${err.message}`);
    } finally {
      askButton.disabled = false;
      askButton.innerHTML = `<span class="btn-text">ASK</span><span class="btn-icon">⚡</span>`;
    }
  }

  // Display Answer & Evidence
  function displayAnswer(data) {
    answerPlaceholder.style.display = "none";
    answerCard.style.display = "block";

    answerText.textContent = data.answer || "No description generated.";
    currentAnswerStart = data.start_time || 0.0;
    answerTimeRange.textContent = data.time_formatted || formatRange(data.start_time, data.end_time);

    // Key objects tags
    if (data.key_objects && data.key_objects.length > 0) {
      keyObjectsContainer.style.display = "flex";
      objectsTagsList.innerHTML = data.key_objects
        .map((obj) => `<span class="object-tag">${obj}</span>`)
        .join("");
    } else {
      keyObjectsContainer.style.display = "none";
    }

    // Evidence frames
    evidenceFramesGrid.innerHTML = "";
    const frames = data.evidence_frames || [];

    if (frames.length > 0) {
      frames.forEach((f) => {
        const card = document.createElement("div");
        card.className = "evidence-frame-card";
        card.innerHTML = `
          <img src="${f.url}" alt="Evidence at ${f.timestamp}s" loading="lazy">
          <span class="frame-time-tag">${f.time_formatted || f.timestamp + 's'}</span>
        `;
        card.addEventListener("click", () => {
          videoPlayer.currentTime = f.timestamp;
          videoPlayer.play();
        });
        evidenceFramesGrid.appendChild(card);
      });
    } else {
      evidenceFramesGrid.innerHTML = `
        <span style="color:var(--text-muted);font-size:0.8rem;grid-column:1/-1;">
          Timestamp window: ${data.start_time}s — ${data.end_time}s. Jump to play evidence.
        </span>
      `;
    }

    // Smooth scroll to answer
    answerCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Jump to Event Button
  jumpToEventBtn.addEventListener("click", () => {
    videoPlayer.currentTime = currentAnswerStart;
    videoPlayer.play();
  });

  // Suggestion Pills Click
  pillButtons.forEach((pill) => {
    pill.addEventListener("click", () => {
      questionInput.value = pill.dataset.query;
      handleAskQuestion();
    });
  });

  // Ask Button & Enter Key
  askButton.addEventListener("click", handleAskQuestion);
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleAskQuestion();
    }
  });

  // Process Video Button
  if (processVideoBtn) {
    processVideoBtn.addEventListener("click", triggerProcessVideo);
  }

  // Initial Load
  loadVideos();
});
