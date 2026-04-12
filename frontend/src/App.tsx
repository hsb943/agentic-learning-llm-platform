import { FormEvent, useMemo, useState } from "react";
import {
  confirmCheckpoints,
  getCurrentQuestion,
  startSession,
  submitAnswer,
} from "./api";
import {
  LearningCheckpoint,
  SessionStateResponse,
  VerificationResult,
} from "./types";

const defaultTopic = "Transformer Neural Networks";
const defaultGoals = "Master transformer architecture and inference flow";
const defaultNote =
  "Transformers use self-attention to relate tokens across a sequence. Positional encoding preserves token order, encoder blocks build contextual representations, and decoder blocks generate outputs autoregressively.";

function readCheckpoints(state: SessionStateResponse): LearningCheckpoint[] {
  const checkpoints = state.values.checkpoints as
    | { checkpoints?: LearningCheckpoint[] }
    | undefined;
  return checkpoints?.checkpoints ?? [];
}

function readVerification(state: SessionStateResponse): VerificationResult | null {
  return (state.values.verifications as VerificationResult | undefined) ?? null;
}

function readSuggestions(state: SessionStateResponse): string[] {
  return readVerification(state)?.suggestions ?? [];
}

export default function App() {
  const [topic, setTopic] = useState(defaultTopic);
  const [goalsText, setGoalsText] = useState(defaultGoals);
  const [note, setNote] = useState(defaultNote);

  const [threadId, setThreadId] = useState<string>("");
  const [status, setStatus] = useState<string>("idle");
  const [checkpoints, setCheckpoints] = useState<LearningCheckpoint[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [feedbackState, setFeedbackState] = useState<SessionStateResponse | null>(
    null,
  );
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);

  const goals = useMemo(
    () =>
      goalsText
        .split(/\r?\n|,/)
        .map((goal) => goal.trim())
        .filter(Boolean),
    [goalsText],
  );

  async function withLoading(task: () => Promise<void>) {
    setIsLoading(true);
    setError("");
    try {
      await task();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  }

  function updateCheckpointField(
    checkpointIndex: number,
    field: keyof LearningCheckpoint,
    value: string | string[],
  ) {
    setCheckpoints((previous) =>
      previous.map((checkpoint, index) =>
        index === checkpointIndex
          ? { ...checkpoint, [field]: value }
          : checkpoint,
      ),
    );
  }

  function updateCriterion(
    checkpointIndex: number,
    criterionIndex: number,
    value: string,
  ) {
    setCheckpoints((previous) =>
      previous.map((checkpoint, index) => {
        if (index !== checkpointIndex) return checkpoint;
        const nextCriteria = checkpoint.criteria.map((criterion, cIndex) =>
          cIndex === criterionIndex ? value : criterion,
        );
        return { ...checkpoint, criteria: nextCriteria };
      }),
    );
  }

  function addCriterion(checkpointIndex: number) {
    setCheckpoints((previous) =>
      previous.map((checkpoint, index) =>
        index === checkpointIndex
          ? { ...checkpoint, criteria: [...checkpoint.criteria, ""] }
          : checkpoint,
      ),
    );
  }

  async function handleStart(event: FormEvent) {
    event.preventDefault();
    await withLoading(async () => {
      const state = await startSession({ topic, goals, note });
      setThreadId(state.thread_id);
      setStatus(state.status);
      setCheckpoints(readCheckpoints(state));
      setCurrentQuestion("");
      setAnswer("");
      setFeedbackState(null);
    });
  }

  async function handleConfirmCheckpoints() {
    if (!threadId) return;
    await withLoading(async () => {
      const state = await confirmCheckpoints(threadId, checkpoints);
      setStatus(state.status);
      setFeedbackState(null);
      const question = await getCurrentQuestion(threadId);
      setCurrentQuestion(question.question);
      setStatus(question.status);
    });
  }

  async function handleSubmitAnswer(event: FormEvent) {
    event.preventDefault();
    if (!threadId) return;
    await withLoading(async () => {
      const state = await submitAnswer(threadId, answer);
      setFeedbackState(state);
      setStatus(state.status);
      setAnswer("");
    });
  }

  const verification = feedbackState ? readVerification(feedbackState) : null;
  const suggestions = feedbackState ? readSuggestions(feedbackState) : [];

  return (
    <div className="page-shell">
      <main className="app-frame">
        <section className="hero-card">
          <p className="eyebrow">Learning Session Builder</p>
          <h1>Study a note, review checkpoints, and answer guided questions.</h1>
          <p className="hero-copy">
            This frontend talks to the FastAPI backend and drives the extracted
            notebook workflow without any notebook widgets.
          </p>
        </section>

        <section className="panel">
          <div className="section-heading">
            <h2>Start Session</h2>
            <span className="status-pill">{status}</span>
          </div>
          <form className="stack" onSubmit={handleStart}>
            <label className="field">
              <span>Topic</span>
              <input
                value={topic}
                onChange={(event) => setTopic(event.target.value)}
                placeholder="Transformer Neural Networks"
              />
            </label>

            <label className="field">
              <span>Goals</span>
              <textarea
                value={goalsText}
                onChange={(event) => setGoalsText(event.target.value)}
                rows={3}
                placeholder="One goal per line or comma-separated"
              />
            </label>

            <label className="field">
              <span>Note</span>
              <textarea
                value={note}
                onChange={(event) => setNote(event.target.value)}
                rows={8}
                placeholder="Paste the study note here"
              />
            </label>

            <div className="actions">
              <button className="primary-button" disabled={isLoading}>
                {isLoading ? "Starting..." : "Start Session"}
              </button>
              {threadId ? <code>thread_id: {threadId}</code> : null}
            </div>
          </form>
        </section>

        {checkpoints.length > 0 ? (
          <section className="panel">
            <div className="section-heading">
              <h2>Checkpoint Review</h2>
              <span>{checkpoints.length} checkpoints</span>
            </div>

            <div className="stack large-gap">
              {checkpoints.map((checkpoint, checkpointIndex) => (
                <article className="checkpoint-card" key={checkpointIndex}>
                  <header className="checkpoint-header">
                    <h3>Checkpoint {checkpointIndex + 1}</h3>
                  </header>

                  <label className="field">
                    <span>Description</span>
                    <textarea
                      rows={3}
                      value={checkpoint.description}
                      onChange={(event) =>
                        updateCheckpointField(
                          checkpointIndex,
                          "description",
                          event.target.value,
                        )
                      }
                    />
                  </label>

                  <div className="criteria-block">
                    <div className="criteria-header">
                      <span>Criteria</span>
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => addCriterion(checkpointIndex)}
                      >
                        Add Criterion
                      </button>
                    </div>

                    {checkpoint.criteria.map((criterion, criterionIndex) => (
                      <input
                        key={criterionIndex}
                        value={criterion}
                        onChange={(event) =>
                          updateCriterion(
                            checkpointIndex,
                            criterionIndex,
                            event.target.value,
                          )
                        }
                        placeholder={`Criterion ${criterionIndex + 1}`}
                      />
                    ))}
                  </div>

                  <label className="field">
                    <span>Verification</span>
                    <textarea
                      rows={3}
                      value={checkpoint.verification}
                      onChange={(event) =>
                        updateCheckpointField(
                          checkpointIndex,
                          "verification",
                          event.target.value,
                        )
                      }
                    />
                  </label>
                </article>
              ))}
            </div>

            <div className="actions">
              <button
                type="button"
                className="primary-button"
                disabled={isLoading || !threadId}
                onClick={handleConfirmCheckpoints}
              >
                {isLoading ? "Confirming..." : "Confirm Checkpoints"}
              </button>
            </div>
          </section>
        ) : null}

        {currentQuestion ? (
          <section className="panel">
            <div className="section-heading">
              <h2>Current Question</h2>
            </div>
            <div className="question-card">
              <p>{currentQuestion}</p>
            </div>

            <form className="stack" onSubmit={handleSubmitAnswer}>
              <label className="field">
                <span>Your Answer</span>
                <textarea
                  rows={6}
                  value={answer}
                  onChange={(event) => setAnswer(event.target.value)}
                  placeholder="Type your answer for the transformer question here"
                />
              </label>

              <div className="actions">
                <button
                  className="primary-button"
                  disabled={isLoading || !answer.trim()}
                >
                  {isLoading ? "Submitting..." : "Submit Answer"}
                </button>
              </div>
            </form>
          </section>
        ) : null}

        {verification ? (
          <section className="panel">
            <div className="section-heading">
              <h2>Feedback</h2>
              <span>
                Understanding: {(verification.understanding_level * 100).toFixed(0)}
                %
              </span>
            </div>

            <div className="feedback-grid">
              <article className="feedback-card">
                <h3>Assessment</h3>
                <p>{verification.feedback}</p>
              </article>

              <article className="feedback-card">
                <h3>Context Alignment</h3>
                <p>{verification.context_alignment ? "Aligned" : "Needs work"}</p>
              </article>
            </div>

            {suggestions.length > 0 ? (
              <article className="feedback-card">
                <h3>Suggestions</h3>
                <ul>
                  {suggestions.map((suggestion) => (
                    <li key={suggestion}>{suggestion}</li>
                  ))}
                </ul>
              </article>
            ) : null}
          </section>
        ) : null}

        {error ? <div className="error-banner">{error}</div> : null}
      </main>
    </div>
  );
}
