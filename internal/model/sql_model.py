create_vacancy_table = """
CREATE TABLE IF NOT EXISTS vacancies(
    id SERIAL PRIMARY KEY,

    name TEXT NOT NULL,
    tags TEXT[] NOT NULL,
    description TEXT NOT NULL,
    red_flags TEXT NOT NULL,
    skill_lvl TEXT NOT NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_vacancy_questions_table = """
CREATE TABLE IF NOT EXISTS vacancy_questions(
    id SERIAL PRIMARY KEY,
    vacancy_id INTEGER NOT NULL REFERENCES vacancies(id) ON DELETE CASCADE,
    
    question TEXT NOT NULL,
    hint_for_evaluation TEXT NOT NULL,
    weight INTEGER NOT NULL,
    question_type TEXT NOT NULL,
    response_time INTEGER NOT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_interview_weights_table = """
CREATE TABLE IF NOT EXISTS interview_weights(
    id SERIAL PRIMARY KEY,
    vacancy_id INTEGER NOT NULL REFERENCES vacancies(id) ON DELETE CASCADE,
    
    logic_structure_score_weight INTEGER NOT NULL,
    soft_skill_score_weight INTEGER NOT NULL,
    hard_skill_score_weight INTEGER NOT NULL,
    accordance_xp_resume_score_weight INTEGER NOT NULL,
    accordance_skill_resume_score_weight INTEGER NOT NULL,
    red_flag_score_weight INTEGER NOT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_resume_weights_table = """
CREATE TABLE IF NOT EXISTS resume_weights(
    id SERIAL PRIMARY KEY,
    vacancy_id INTEGER NOT NULL REFERENCES vacancies(id) ON DELETE CASCADE,
    
    accordance_xp_vacancy_score_threshold INTEGER NOT NULL,
    accordance_skill_vacancy_score_threshold INTEGER NOT NULL,
    recommendation_weight INTEGER NOT NULL,
    portfolio_weight INTEGER NOT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_interviews_table = """
CREATE TABLE IF NOT EXISTS interviews(
    id SERIAL PRIMARY KEY,
    vacancy_id INTEGER NOT NULL REFERENCES vacancies(id) ON DELETE CASCADE,
    
    candidate_email TEXT NOT NULL,
    candidate_name TEXT NOT NULL,
    candidate_phone TEXT NOT NULL,
    candidate_telegram_login TEXT NOT NULL,
    candidate_resume_fid TEXT NOT NULL,
    candidate_resume_filename TEXT NOT NULL,
    accordance_xp_vacancy_score INTEGER NOT NULL,
    accordance_skill_vacancy_score INTEGER NOT NULL,
    
    red_flag_score INTEGER DEFAULT 0,
    hard_skill_score INTEGER DEFAULT 0, 
    soft_skill_score INTEGER DEFAULT 0,
    logic_structure_score INTEGER DEFAULT 0,
    accordance_xp_resume_score INTEGER DEFAULT 0,
    accordance_skill_resume_score INTEGER DEFAULT 0,
    strong_areas TEXT DEFAULT '',
    weak_areas TEXT DEFAULT '',
    approved_skills TEXT[] DEFAULT '{}',
    
    general_score INTEGER DEFAULT 0,
    general_result TEXT NOT NULL,
    message_to_candidate TEXT NOT NULL DEFAULT '',
    message_to_hr TEXT NOT NULL DEFAULT '',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_candidate_answers_table = """
CREATE TABLE IF NOT EXISTS candidate_answers(
    id SERIAL PRIMARY KEY,
    question_id INTEGER NOT NULL REFERENCES vacancy_questions(id) ON DELETE CASCADE,
    interview_id INTEGER NOT NULL REFERENCES interviews(id) ON DELETE CASCADE,
    
    response_time INTEGER DEFAULT 0,
    message_ids INTEGER[] DEFAULT '{}',
    message_to_candidate TEXT DEFAULT '',
    message_to_hr TEXT DEFAULT '',
    score INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

create_interview_messages_table = """
CREATE TABLE IF NOT EXISTS interview_messages(
    id SERIAL PRIMARY KEY,
    interview_id INTEGER NOT NULL REFERENCES interviews(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES vacancy_questions(id) ON DELETE CASCADE,
    
    audio_fid TEXT NOT NULL,
    audio_name TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

drop_vacancy_table = """
DROP TABLE IF EXISTS vacancies CASCADE;
"""

drop_vacancy_questions_table = """
DROP TABLE IF EXISTS vacancy_questions CASCADE;
"""

drop_interview_weights_table = """
DROP TABLE IF EXISTS interview_weights CASCADE;
"""

drop_resume_weights_table = """
DROP TABLE IF EXISTS resume_weights CASCADE;
"""

drop_interviews_table = """
DROP TABLE IF EXISTS interviews CASCADE;
"""

drop_candidate_answers_table = """
DROP TABLE IF EXISTS candidate_answers CASCADE;
"""

drop_interview_messages_table = """
DROP TABLE IF EXISTS interview_messages CASCADE;
"""

create_all_tables_queries = [
    create_vacancy_table,
    create_vacancy_questions_table,
    create_interviews_table,
    create_interview_weights_table,
    create_resume_weights_table,
    create_candidate_answers_table,
    create_interview_messages_table,
]


drop_all_tables_queries = [
    drop_candidate_answers_table,
    drop_interview_weights_table,
    drop_resume_weights_table,
    drop_interviews_table,
    drop_vacancy_questions_table,
    drop_interview_messages_table,
    drop_vacancy_table,
]