export type LearningCheckpoint = {
  description: string;
  criteria: string[];
  verification: string;
};

export type SessionStateResponse = {
  thread_id: string;
  status: string;
  next_nodes: string[];
  values: Record<string, unknown>;
};

export type QuestionResponse = {
  thread_id: string;
  current_checkpoint: number;
  question: string;
  status: string;
};

export type VerificationResult = {
  understanding_level: number;
  feedback: string;
  suggestions: string[];
  context_alignment: boolean;
};
