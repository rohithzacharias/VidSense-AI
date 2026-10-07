I want to create the FRONTEND part of my Hacknex-VidSense AI project.

IMPORTANT:
ONLY work on the frontend in this task.

Use:
- React
- Vite
- JavaScript
- Tailwind CSS
- Axios or fetch

DO NOT use TypeScript.

Do NOT create:
- .ts files
- .tsx files
- tsconfig.json
- TypeScript types/interfaces

Everything in the frontend must use normal JavaScript and JSX.

==================================================
PROJECT CONTEXT
==================================================

The project is:

Hacknex-VidSense AI

It is a Video Understanding & Temporal Reasoning system.

The complete system will eventually be:

VIDEO
 ↓
Frame extraction
 ↓
YOLO
 ↓
BoT-SORT
 ↓
Qwen2.5-VL
 ↓
Event generation
 ↓
Embeddings
 ↓
PostgreSQL + pgvector
 ↓
Retriever
 ↓
Temporal reasoning
 ↓
Final LLM
 ↓
Answer + timestamp
 ↓
Frontend

For THIS TASK, only build the frontend.

The backend/AI pipeline is being developed separately.

==================================================
DO NOT TOUCH BACKEND
==================================================

Do NOT modify or implement:

- YOLO
- BoT-SORT
- Qwen2.5-VL
- video processing
- embeddings
- PostgreSQL
- pgvector
- RAG
- retriever
- temporal reasoning backend
- final LLM
- FastAPI backend
- authentication
- Docker

The frontend must work independently using mock data.

==================================================
FRONTEND LOCATION
==================================================

Create:

frontend/

inside the existing project.

Do NOT create a separate Git repository inside frontend.

The main project repository should remain:

Hacknex-VidSense AI/

==================================================
TECH STACK
==================================================

Use exactly:

React
Vite
JavaScript
JSX
Tailwind CSS

Use normal JavaScript.

Do NOT use TypeScript.

Use fetch or Axios for future API communication.

Keep dependencies minimal.

Do not use Next.js.

Do not use Redux unless absolutely necessary.

==================================================
FOLDER STRUCTURE
==================================================

Create:

frontend/
│
├── public/
│
├── src/
│   │
│   ├── components/
│   │   ├── Navbar/
│   │   │   └── Navbar.jsx
│   │   │
│   │   ├── VideoUploader/
│   │   │   └── VideoUploader.jsx
│   │   │
│   │   ├── VideoPlayer/
│   │   │   └── VideoPlayer.jsx
│   │   │
│   │   ├── QuestionBox/
│   │   │   └── QuestionBox.jsx
│   │   │
│   │   ├── AnswerCard/
│   │   │   └── AnswerCard.jsx
│   │   │
│   │   ├── EvidenceList/
│   │   │   └── EvidenceList.jsx
│   │   │
│   │   ├── Timeline/
│   │   │   └── Timeline.jsx
│   │   │
│   │   ├── EventItem/
│   │   │   └── EventItem.jsx
│   │   │
│   │   ├── LoadingState/
│   │   │   └── LoadingState.jsx
│   │   │
│   │   └── ErrorMessage/
│   │       └── ErrorMessage.jsx
│   │
│   ├── pages/
│   │   └── Dashboard/
│   │       └── Dashboard.jsx
│   │
│   ├── services/
│   │   ├── api.js
│   │   └── videoApi.js
│   │
│   ├── mock/
│   │   ├── events.js
│   │   └── responses.js
│   │
│   ├── hooks/
│   │   └── useVideoPlayer.js
│   │
│   ├── utils/
│   │   └── formatTimestamp.js
│   │
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
│
├── README.md
├── package.json
├── vite.config.js
├── .env.example
├── .gitignore
└── index.html

Everything must be JavaScript/JSX.

==================================================
APPLICATION UI
==================================================

Create a modern video-analysis dashboard.

The main page should contain:

1. Header
2. Video upload
3. Video player
4. Current timestamp
5. Question input
6. Ask button
7. Answer
8. Answer timestamp
9. Evidence
10. Event timeline

The UI should feel like a professional AI video analysis product.

Use a dark, modern interface.

Keep it clean.

Do not make it overly flashy.

==================================================
HEADER
==================================================

Display:

Hacknex VidSense AI

Subtitle:

"Understand what happened in your videos."

==================================================
VIDEO UPLOAD
==================================================

Create a video upload component.

Example:

┌─────────────────────────────────────┐
│                                     │
│       Drop video here               │
│                                     │
│             or                      │
│                                     │
│        [ Choose Video ]             │
│                                     │
└─────────────────────────────────────┘

The user should be able to select a local video.

After selecting:

- show the video in the video player
- show filename
- show duration if available

For now, local video preview is enough.

Do NOT require the backend to preview a video.

==================================================
VIDEO PLAYER
==================================================

Use the HTML5 <video> element.

The video player must support:

- play
- pause
- seek
- volume
- fullscreen
- current time
- duration

The application must also be able to programmatically seek to a timestamp.

For example:

video.currentTime = 18

The parent component should be able to tell the video player:

"Jump to 18 seconds."

==================================================
CURRENT TIMESTAMP
==================================================

Display:

00:18 / 02:35

Update it while the video is playing.

Create a reusable utility:

formatTimestamp(seconds)

Examples:

0 → 00:00
5 → 00:05
65 → 01:05
125 → 02:05

For longer videos:

01:02:35

==================================================
QUESTION BOX
==================================================

Create:

Ask about this video

Textarea/input:

"What happened before the safety alarm?"

Button:

[ Ask ]

When the user clicks Ask:

- show loading state
- call the API service
- display answer

Initially use mock data.

==================================================
ANSWER CARD
==================================================

Example:

ANSWER

Person #7 opened the machine panel immediately
before the safety alarm.

⏱ 00:18 – 00:20

[ ▶ Jump to event ]

The button must seek the video to start_time.

==================================================
EVIDENCE
==================================================

Display the events used to generate the answer.

Example:

EVIDENCE

00:12 – 00:15
Person #7 approaches the machine.

00:18 – 00:20
Person #7 opens the machine panel.

00:25 – 00:26
Safety alarm activates.

Every evidence item should be clickable.

Clicking it should seek the video to its start_time.

==================================================
EVENT TIMELINE
==================================================

Create a timeline/list of all events.

Example:

EVENT TIMELINE

● 00:05
  Person #7 enters restricted area.

● 00:12
  Person #7 approaches machine.

● 00:18
  Person #7 opens machine panel.

● 00:25
  Safety alarm activates.

● 00:31
  Person #7 leaves area.

Clicking an event should:

1. Select the event
2. Highlight it
3. Seek the video to the event's start_time

If possible, visually indicate the event currently occurring based on video.currentTime.

==================================================
DATA STRUCTURE
==================================================

Use normal JavaScript objects.

Event example:

{
    event_id: "evt_001",
    video_id: "video_001",
    start_time: 5,
    end_time: 10,
    event_type: "entry",
    description: "Person #7 enters the restricted area.",
    track_ids: [7],
    confidence: 0.93
}

Answer example:

{
    answer:
        "Person #7 opened the machine panel immediately before the safety alarm.",
    start_time: 18,
    end_time: 20,
    evidence: [...]
}

Video example:

{
    video_id: "video_001",
    filename: "factory.mp4",
    duration: 120
}

Do NOT create TypeScript interfaces.

==================================================
MOCK DATA
==================================================

The frontend MUST work even when the backend is completely offline.

Create:

src/mock/events.js

and:

src/mock/responses.js

Use this event data:

export const mockEvents = [
    {
        event_id: "evt_001",
        video_id: "video_001",
        start_time: 5,
        end_time: 10,
        event_type: "entry",
        description: "Person #7 enters the restricted area.",
        track_ids: [7],
        confidence: 0.93
    },
    {
        event_id: "evt_002",
        video_id: "video_001",
        start_time: 12,
        end_time: 15,
        event_type: "movement",
        description: "Person #7 approaches the machine.",
        track_ids: [7],
        confidence: 0.91
    },
    {
        event_id: "evt_003",
        video_id: "video_001",
        start_time: 18,
        end_time: 20,
        event_type: "interaction",
        description: "Person #7 opens the machine panel.",
        track_ids: [7],
        confidence: 0.92
    },
    {
        event_id: "evt_004",
        video_id: "video_001",
        start_time: 25,
        end_time: 26,
        event_type: "alarm",
        description: "Safety alarm activates.",
        track_ids: [],
        confidence: 0.96
    },
    {
        event_id: "evt_005",
        video_id: "video_001",
        start_time: 31,
        end_time: 35,
        event_type: "exit",
        description: "Person #7 leaves the restricted area.",
        track_ids: [7],
        confidence: 0.90
    }
];

Create a mock response:

export const mockAnswer = {
    answer:
        "Person #7 opened the machine panel immediately before the safety alarm.",
    start_time: 18,
    end_time: 20,
    evidence: mockEvents.slice(1, 4)
};

==================================================
API SERVICE
==================================================

Create:

src/services/api.js

and:

src/services/videoApi.js

Components must NOT directly call fetch or axios.

All API communication must go through the service layer.

Create functions such as:

uploadVideo(file)

getVideoEvents(videoId)

askQuestion(videoId, question)

For now, these functions should support mock mode.

==================================================
MOCK / REAL API MODE
==================================================

Use:

VITE_USE_MOCK_API=true

in .env.example.

When:

VITE_USE_MOCK_API=true

use mock data.

When:

VITE_USE_MOCK_API=false

use the future FastAPI backend.

==================================================
FUTURE BACKEND API
==================================================

The frontend should be designed around this API.

BASE URL:

VITE_API_BASE_URL=http://localhost:8000

--------------------------------------------------

POST /api/videos/upload

Request:

multipart/form-data

Field:

video

Response:

{
    "video_id": "video_001",
    "filename": "factory.mp4",
    "duration": 120
}

--------------------------------------------------

GET /api/videos/{video_id}/events

Response:

{
    "video_id": "video_001",
    "events": [...]
}

--------------------------------------------------

POST /api/videos/{video_id}/query

Request:

{
    "question": "What happened before the safety alarm?"
}

Response:

{
    "answer": "Person #7 opened the machine panel immediately before the safety alarm.",
    "start_time": 18,
    "end_time": 20,
    "evidence": [...]
}

The frontend must be ready to switch from mock API to real API without changing the UI components.

==================================================
VIDEO PLAYER STATE
==================================================

Create:

src/hooks/useVideoPlayer.js

Manage:

- video reference
- currentTime
- duration
- playing state
- seekTo(timestamp)

The important function is:

seekTo(seconds)

It should make the video jump to that exact timestamp.

==================================================
COMPONENT RESPONSIBILITIES
==================================================

VideoUploader.jsx

Responsible only for:
- selecting video
- validating basic video file
- returning selected file

VideoPlayer.jsx

Responsible only for:
- rendering video
- playback
- current time
- seeking

QuestionBox.jsx

Responsible only for:
- question input
- Ask button
- loading state

AnswerCard.jsx

Responsible only for:
- displaying answer
- displaying timestamp
- jump-to-event button

EvidenceList.jsx

Responsible only for:
- rendering evidence events

Timeline.jsx

Responsible only for:
- rendering complete event timeline
- selected event

EventItem.jsx

Responsible only for:
- rendering a single event
- handling click

Do not put the entire application into App.jsx.

==================================================
STATE
==================================================

Keep state simple.

The Dashboard can manage:

- selected video
- video URL
- events
- currentTime
- selectedEvent
- question
- answer
- loading
- error

Do not introduce Redux.

React state/hooks are sufficient.

==================================================
DESIGN REQUIREMENTS
==================================================

Use Tailwind CSS.

Theme:

Dark professional AI dashboard.

Suggested structure:

┌───────────────────────────────────────────────────────┐
│ Hacknex VidSense AI                                   │
│ Understand what happened in your videos.             │
├───────────────────────────────────────────────────────┤
│                                                       │
│                 VIDEO PLAYER                          │
│                                                       │
├───────────────────────────────────────────────────────┤
│                                                       │
│ Ask about this video                                  │
│                                                       │
│ ┌───────────────────────────────────────────────────┐ │
│ │ What happened before the safety alarm?           │ │
│ └───────────────────────────────────────────────────┘ │
│                           [ ASK ]                     │
│                                                       │
├───────────────────────────────────────────────────────┤
│ ANSWER                                                │
│                                                       │
│ Person #7 opened the machine panel...                │
│                                                       │
│ 00:18 – 00:20       [ Jump to event ]               │
│                                                       │
├───────────────────────────────────────────────────────┤
│ EVIDENCE                                              │
│                                                       │
│ 00:12 Person approaches machine                       │
│ 00:18 Person opens machine panel                      │
│ 00:25 Safety alarm activates                          │
│                                                       │
├───────────────────────────────────────────────────────┤
│ EVENT TIMELINE                                        │
│                                                       │
│ ● 00:05 Person enters restricted area                │
│ ● 00:12 Person approaches machine                    │
│ ● 00:18 Person opens machine panel                   │
│ ● 00:25 Safety alarm activates                       │
│ ● 00:31 Person leaves area                            │
│                                                       │
└───────────────────────────────────────────────────────┘

Use responsive layouts.

==================================================
LOADING STATES
==================================================

Create reusable loading UI.

Examples:

"Uploading video..."

"Analyzing video..."

"Finding relevant events..."

"Generating answer..."

The UI must remain responsive.

==================================================
ERROR HANDLING
==================================================

Create a reusable error component.

Friendly messages:

"Unable to load this video."

"Unable to analyze the video."

"Unable to get an answer."

Do not expose raw errors or stack traces.

==================================================
README
==================================================

Create:

frontend/README.md

The README must explain clearly:

1. What Hacknex VidSense AI is
2. What the frontend does
3. Technology stack
4. Folder structure
5. Installation
6. Running the frontend
7. Mock mode
8. Real API mode
9. API contract
10. Components
11. Data structures
12. Timestamp seeking
13. How to add a new component
14. How teammates should work with Git
15. Current limitations

Explain that the backend/AI pipeline is separate.

==================================================
GIT
==================================================

Do NOT initialize a nested Git repository inside frontend.

The main repository is:

Hacknex-VidSense AI/

The frontend is just:

Hacknex-VidSense AI/frontend/

Create:

frontend/.gitignore

Ignore:

node_modules/
dist/
.env
.env.local

Keep:

.env.example

==================================================
TEAM WORKFLOW
==================================================

Document example branches:

feature/frontend-upload
feature/frontend-video-player
feature/frontend-question
feature/frontend-timeline
feature/frontend-answer
feature/frontend-ui

Keep commits focused.

Example:

feat(frontend): add video uploader

feat(frontend): add timeline component

feat(frontend): add question interface

==================================================
RUNNING
==================================================

The frontend must run with:

cd frontend
npm install
npm run dev

It must work with:

VITE_USE_MOCK_API=true

without any backend running.

==================================================
IMPORTANT
==================================================

Do NOT use TypeScript.

Do NOT create:

.ts
.tsx
tsconfig.json

Use only:

.js
.jsx
.css

The frontend must be completely independent from the Python backend.

Do not modify the existing AI/backend pipeline.

==================================================
FINAL REPORT
==================================================

After implementation, report:

1. Files created
2. Files modified
3. npm packages installed
4. Exact command to run frontend
5. Mock mode instructions
6. Real API mode instructions
7. API contract
8. Components created
9. Git workflow
10. Any issues

Do not just say "frontend created".

Verify that:

npm install

and:

npm run dev

work successfully.

The final frontend should be usable as a standalone React + JavaScript application using mock data.