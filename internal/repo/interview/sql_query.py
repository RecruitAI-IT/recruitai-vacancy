create_interview = """
INSERT INTO interviews (
    vacancy_id,
    candidate_name,
    candidate_email,
    candidate_phone,
    candidate_telegram_login,
    candidate_resume_fid,
    candidate_resume_filename,
    accordance_xp_vacancy_score,
    accordance_skill_vacancy_score,
    general_result
)
VALUES (
    :vacancy_id,
    :candidate_name,
    :candidate_email,
    :candidate_phone,
    :candidate_telegram_login,
    :candidate_resume_fid,
    :candidate_resume_filename,
    :accordance_xp_vacancy_score,
    :accordance_skill_vacancy_score,
    :general_result
)
RETURNING id;
"""

fill_interview_criterion = """
UPDATE interviews
SET 
    red_flag_score = :red_flag_score,
    hard_skill_score = :hard_skill_score,
    soft_skill_score = :soft_skill_score,
    logic_structure_score = :logic_structure_score,
    accordance_xp_resume_score = :accordance_xp_resume_score,
    accordance_skill_resume_score = :accordance_skill_resume_score,
    strong_areas = :strong_areas,
    weak_areas = :weak_areas,
    approved_skills = :approved_skills,
    general_score = :general_score,
    general_result = :general_result,
    message_to_candidate = :message_to_candidate,
    message_to_hr = :message_to_hr
WHERE id = :interview_id;
"""

create_interview_message = """
INSERT INTO interview_messages (
    interview_id,
    question_id,
    audio_name,
    audio_fid,
    role,
    text
)
VALUES (
    :interview_id,
    :question_id,
    :audio_name,
    :audio_fid,
    :role,
    :text
)
RETURNING id;
"""

create_candidate_answer = """
INSERT INTO candidate_answers (
    question_id,
    interview_id
)
VALUES (
    :question_id,
    :interview_id
)
RETURNING id;
"""

add_message_to_candidate_answer = """
UPDATE candidate_answers
SET message_ids = array_append(message_ids, :message_id)
WHERE id = :candidate_answer_id;
"""

evaluation_candidate_answer = """
UPDATE candidate_answers
SET 
    score = :score,
    message_to_candidate = :message_to_candidate,
    message_to_hr = :message_to_hr,
    response_time = :response_time
WHERE id = :candidate_answer_id;
"""

get_candidate_answer = """
SELECT * FROM candidate_answers
WHERE question_id = :question_id AND interview_id = :interview_id;
"""

get_interview_by_id = """
SELECT * FROM interviews
WHERE id = :interview_id;
"""

get_all_interview = """
SELECT * FROM interviews
WHERE vacancy_id = :vacancy_id
ORDER BY created_at DESC;
"""

get_all_candidate_answer = """
SELECT * FROM candidate_answers
WHERE interview_id = :interview_id
ORDER BY question_id;
"""

get_interview_messages = """
SELECT * FROM interview_messages
WHERE interview_id = :interview_id
ORDER BY created_at;
"""