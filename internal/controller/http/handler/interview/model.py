from pydantic import BaseModel
from internal import model


class StartInterviewResponse(BaseModel):
    message_to_candidate: str
    total_question: int
    question_id: int
    llm_audio_filename: str
    llm_audio_fid: str


class SendAnswerResponse(BaseModel):
    question_id: int
    message_to_candidate: str
    interview_result: dict
    llm_audio_filename: str
    llm_audio_fid: str


class GetCandidateAnswersResponse(BaseModel):
    candidate_answers: list[model.CandidateAnswer]
    interview_messages: list[model.InterviewMessage]
