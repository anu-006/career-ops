"""
Career-Ops Platform - Machine Learning Skill Prediction & Gap Analysis Service
=============================================================================
Path: career-ops/ml/skill-prediction/career_ops_skill_service.py

This module provides production-grade inference functions for Career-Ops.
It loads pre-trained TF-IDF vectorizers, label encoders, and prediction metadata
to extract skills from job descriptions and identify candidate skill gaps.

Usage:
  # Programmatic (Python):
  from ml.skill_prediction.career_ops_skill_service import predict_skills_from_text, analyze_skill_gap

  # Command Line (CLI):
  python career_ops_skill_service.py --jd "Software engineer with Python and SQL"
  python career_ops_skill_service.py --jd-file ../../jds/acme.md --skills "Python,SQL,Docker"
  python career_ops_skill_service.py --jd-file ../../jds/acme.md --cv-file ../../data/cv.md --json
"""

import os
import sys
import re
import json
import argparse
import joblib
import numpy as np

# Resolve path to saved artifacts relative to this script
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
TFIDF_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
MLB_PATH = os.path.join(MODEL_DIR, "skill_binarizer.joblib")
CLF_PATH = os.path.join(MODEL_DIR, "skill_classifier.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Lazy-loaded artifact globals
_tfidf = None
_mlb = None
_clf = None
_metadata = None
_default_threshold = 0.50


def load_artifacts(verbose=False):
    """Loads serialized vectorizer, label binarizer, classifier, and metadata."""
    global _tfidf, _mlb, _clf, _metadata, _default_threshold

    if _tfidf is None or _mlb is None:
        if not os.path.exists(TFIDF_PATH) or not os.path.exists(MLB_PATH):
            raise FileNotFoundError(
                f"Missing model artifacts in {MODEL_DIR}. Please run training notebook first."
            )

        if verbose:
            print("Loading Career-Ops ML Skill Predictor artifacts...", file=sys.stderr)

        _tfidf = joblib.load(TFIDF_PATH)
        _mlb = joblib.load(MLB_PATH)

        if os.path.exists(CLF_PATH):
            _clf = joblib.load(CLF_PATH)

        if os.path.exists(META_PATH):
            with open(META_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
            _default_threshold = _metadata.get("optimal_threshold", 0.50)

        if verbose:
            print(
                f"Loaded successfully! Vocabulary: {len(_mlb.classes_)} skills, "
                f"Threshold: {_default_threshold}",
                file=sys.stderr
            )

    return _tfidf, _mlb, _clf, _default_threshold


def predict_skills_from_text(job_description, threshold=None, top_k=15, verbose=False):
    """
    Extracts relevant skills and confidence scores from a job description.

    Parameters:
        job_description (str): Raw text of the job description.
        threshold (float): Decision cutoff for activating skill predictions.
        top_k (int): Maximum number of top skills to return.
        verbose (bool): Whether to log loading information.

    Returns:
        list of dict: [{'skill': skill_name, 'confidence': score}, ...]
    """
    if not job_description or not str(job_description).strip():
        return []

    tfidf, mlb, clf, def_threshold = load_artifacts(verbose=verbose)
    cutoff = threshold if threshold is not None else def_threshold

    # 1. Transform raw text using the fitted TF-IDF vocabulary
    vec = tfidf.transform([str(job_description)])

    # 2. If trained classifier is available, predict probabilities
    if clf is not None:
        probs = clf.predict_proba(vec)[0]
        classes = mlb.classes_
        
        # Filter active indices
        active_indices = np.where(probs >= cutoff)[0]
        # Sort descending by probability
        sorted_indices = active_indices[np.argsort(-probs[active_indices])]
        
        predictions = [
            {"skill": classes[i], "confidence": round(float(probs[i]), 4)}
            for i in sorted_indices[:top_k]
        ]
        return predictions

    # Fallback to feature alignment
    feature_names = np.array(tfidf.get_feature_names_out())
    doc_tfidf = vec.toarray()[0]
    matched_features = set(feature_names[doc_tfidf > 0])

    predictions = []
    for cls in mlb.classes_:
        cls_tokens = cls.split()
        if all(token in matched_features for token in cls_tokens):
            token_scores = [doc_tfidf[np.where(feature_names == t)[0][0]] for t in cls_tokens if t in matched_features]
            score = float(np.mean(token_scores)) if token_scores else 0.0
            prob = round(min(0.98, max(0.30, score * 2.5 + 0.20)), 4)
            if prob >= cutoff:
                predictions.append({"skill": cls, "confidence": prob})

    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    return predictions[:top_k]


def extract_skills_from_cv_file(cv_path):
    """Extracts known candidate skills from a CV markdown file."""
    if not os.path.exists(cv_path):
        return []

    with open(cv_path, "r", encoding="utf-8") as f:
        text = f.read()

    candidate_skills = set()
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            items = re.split(r'[,|;]', line.lstrip("- *"))
            for item in items:
                cleaned = item.strip()
                if cleaned and len(cleaned) < 40:
                    candidate_skills.add(cleaned)
    return sorted(list(candidate_skills))


def analyze_skill_gap(job_description, candidate_skills, threshold=None, verbose=False):
    """
    Performs skill gap analysis between an employer's job description
    and a job seeker's resume skills.

    Parameters:
        job_description (str): Text of the job posting.
        candidate_skills (list of str): Skills currently possessed by the candidate.
        threshold (float): Decision threshold for skill prediction.
        verbose (bool): Whether to print debug information.

    Returns:
        dict: Detailed breakdown of matched skills, missing skills, and match percentage.
    """
    # 1. Extract target skills from the job description
    predicted_records = predict_skills_from_text(job_description, threshold=threshold, verbose=verbose)
    predicted_skills_set = {r["skill"] for r in predicted_records}

    # 2. Normalize candidate skills (handling hyphens and whitespace consistently)
    normalized_candidate = set()
    for s in candidate_skills:
        if s and str(s).strip():
            cleaned = re.sub(r'\s+', ' ', str(s).strip().lower())
            normalized_candidate.add(cleaned)
            normalized_candidate.add(cleaned.replace("-", " "))

    # 3. Perform Set Operations
    matched = sorted(list(predicted_skills_set & normalized_candidate))
    missing = sorted(list(predicted_skills_set - normalized_candidate))

    # 4. Calculate Match Percentage
    if predicted_skills_set:
        match_percentage = round((len(matched) / len(predicted_skills_set)) * 100.0, 2)
    else:
        match_percentage = 0.0

    return {
        "job_predicted_skills": [r["skill"] for r in predicted_records],
        "predicted_with_confidence": predicted_records,
        "candidate_skills": sorted(list(normalized_candidate)),
        "matched_skills": matched,
        "missing_skills": missing,
        "match_percentage": match_percentage,
        "disclaimer": (
            "This prediction and gap analysis is an algorithmic estimation based on learned "
            "patterns from historical job descriptions. It does not certify candidate ability "
            "nor guarantee employer requirements."
        )
    }


def main():
    parser = argparse.ArgumentParser(
        description="Career-Ops Machine Learning Skill Predictor & Gap Analyzer"
    )
    parser.add_argument("--jd", type=str, help="Raw job description text string")
    parser.add_argument("--jd-file", type=str, help="Path to job description text/markdown file")
    parser.add_argument("--skills", type=str, help="Comma-separated list of candidate skills")
    parser.add_argument("--cv-file", type=str, help="Path to candidate cv.md file")
    parser.add_argument("--threshold", type=float, default=None, help="Prediction threshold (default: 0.50)")
    parser.add_argument("--top-k", type=int, default=15, help="Max skills to return (default: 15)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON to stdout")
    parser.add_argument("--self-test", action="store_true", help="Run self-test with sample job description")

    args = parser.parse_args()

    # Self-test mode
    if args.self_test or (len(sys.argv) == 1):
        sample_jd = (
            "Senior Machine Learning Engineer needed with strong Python, SQL, and Apache Spark experience. "
            "You will build deep learning pipelines using TensorFlow and PyTorch, deploy to AWS cloud, "
            "and conduct statistical modeling and A/B testing. Familiarity with Tableau, Pandas, and Scikit-learn required."
        )
        sample_skills = ["Python", "SQL", "Pandas", "Scikit-Learn", "Tableau", "Git"]

        report = analyze_skill_gap(sample_jd, sample_skills, threshold=args.threshold, verbose=not args.json)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("\nCareer-Ops ML Skill Prediction & Gap Analysis (Self-Test)")
            print("=" * 60)
            print(f"Overall Match: {report['match_percentage']}%")
            print(f"\nMatched Skills ({len(report['matched_skills'])}):")
            for s in report['matched_skills']:
                print(f"  [x] {s}")
            print(f"\nMissing Skills ({len(report['missing_skills'])}):")
            for s in report['missing_skills']:
                print(f"  [ ] {s}")
            print("\nTop Predicted Skills with Confidence:")
            for p in report['predicted_with_confidence']:
                print(f"  * {p['skill']:25s} ({p['confidence']*100:.1f}%)")
        return

    # Extract JD text
    jd_text = ""
    if args.jd:
        jd_text = args.jd
    elif args.jd_file:
        if not os.path.exists(args.jd_file):
            print(f"Error: File not found: {args.jd_file}", file=sys.stderr)
            sys.exit(1)
        with open(args.jd_file, "r", encoding="utf-8") as f:
            jd_text = f.read()

    if not jd_text:
        print("Error: No job description provided. Use --jd or --jd-file.", file=sys.stderr)
        sys.exit(1)

    # Extract candidate skills
    candidate_skills = []
    if args.skills:
        candidate_skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    elif args.cv_file:
        candidate_skills = extract_skills_from_cv_file(args.cv_file)

    # Execute gap analysis or simple prediction
    if candidate_skills:
        report = analyze_skill_gap(jd_text, candidate_skills, threshold=args.threshold, verbose=not args.json)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"\nSkill Gap Report (Match: {report['match_percentage']}%)")
            print("-" * 45)
            print(f"Matched ({len(report['matched_skills'])}): {', '.join(report['matched_skills']) or '(none)'}")
            print(f"Missing ({len(report['missing_skills'])}): {', '.join(report['missing_skills']) or '(none)'}")
    else:
        predictions = predict_skills_from_text(
            jd_text, threshold=args.threshold, top_k=args.top_k, verbose=not args.json
        )
        if args.json:
            print(json.dumps({"predicted_skills": predictions}, indent=2))
        else:
            print("\nPredicted Technical Skills:")
            print("-" * 35)
            for p in predictions:
                print(f"  {p['skill']:25s} ({p['confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
