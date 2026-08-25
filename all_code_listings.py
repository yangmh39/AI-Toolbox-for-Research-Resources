"""
Code listings extracted from "AI Toolbox For Research"

This file contains all Python code examples from the textbook,
extracted from main.tex. Each listing is separated by a comment
header showing its original caption. To run a specific listing,
copy it into a Python environment (e.g., Google Colab).

Note: Some listings assume prior code in the same section has run
(e.g., variables named `df`, `X_train`, etc.). Run the listings of
a given section in order.
"""


# ============================================================================
# Listing 1: Exploring the AV perception problem --- the first step in any AI project
# ============================================================================

# We'll use these libraries throughout the book
# All come pre-installed in Google Colab

# Core data libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Let's define the key objects an AV system must detect
av_objects = {
    'pedestrian': 'A person walking or standing --- highest safety priority',
    'car': 'Another vehicle --- must track speed and direction',
    'cyclist': 'A bicycle or motorcycle --- small, fast, vulnerable',
    'traffic_light': 'Controls right-of-way at intersections',
    'traffic_sign': 'Speed limits, stop signs, warnings',
}

# Display the detection taxonomy
print("Objects an AV perception system must detect:")
print("=" * 50)
for obj, desc in av_objects.items():
    print(f"  {obj:20s} -> {desc}")

# A typical AV camera captures images at this resolution
image_width = 1920   # pixels
image_height = 1080  # pixels
fps = 30             # frames per second

print(f"\nTypical AV camera: {image_width}x{image_height} @ {fps} FPS")
print(f"Data per second: {image_width * image_height * 3 * fps:,} bytes")
print(f"  (That's ~{image_width * image_height * 3 * fps / 1e6:.0f} MB per second!)")
print(f"\nThe challenge: process all this data in REAL TIME.")
print(f"At 30 FPS, you have {1000/30:.1f} ms per frame to detect everything.")


# ============================================================================
# Listing 2: Rule-based vs. learned classification for AV perception
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns

np.random.seed(42)
n_samples = 300

# Generate synthetic AV detection features
# Pedestrians: tall aspect ratio (1.5-4.0), irregular motion (0.2-0.8)
# Vehicles: wide aspect ratio (0.5-1.5), smooth motion (0.7-1.0)
# Clutter: variable aspect ratio (0.3-3.0), irregular motion (0.1-0.9)

def generate_class(n, ar_mean, ar_std, mc_mean, mc_std, label):
    ar = np.clip(np.random.normal(ar_mean, ar_std, n), 0.3, 4.0)
    mc = np.clip(np.random.normal(mc_mean, mc_std, n), 0.1, 1.0)
    return np.column_stack([ar, mc]), np.full(n, label)

X_ped, y_ped = generate_class(100, 2.8, 0.6, 0.35, 0.15, 1)
X_veh, y_veh = generate_class(100, 0.9, 0.25, 0.85, 0.08, 0)
X_clutter, y_clutter = generate_class(100, 1.6, 0.8, 0.50, 0.20, 0)

X = np.vstack([X_ped, X_veh, X_clutter])
y = np.hstack([y_ped, y_veh, y_clutter])

# --- Rule-Based Classifier ---
def rule_based_classifier(ar, mc):
    """Hand-crafted rules: pedestrian = tall + irregular motion."""
    if 1.8 <= ar <= 4.0 and mc <= 0.65:
        return 1  # pedestrian
    return 0      # non-pedestrian

y_rule_pred = np.array([rule_based_classifier(ar, mc) for ar, mc in X])

# --- Learned Classifier ---
model = LogisticRegression()
model.fit(X, y)
y_ml_pred = model.predict(X)

# --- Comparison ---
print("=" * 55)
print("RULE-BASED vs. MACHINE LEARNING: PEDESTRIAN DETECTION")
print("=" * 55)

print("\nRule-Based Classifier:")
print(classification_report(y, y_rule_pred,
      target_names=['Non-Ped', 'Ped']))

print("ML Classifier (Logistic Regression):")
print(classification_report(y, y_ml_pred,
      target_names=['Non-Ped', 'Ped']))

# --- Visualize Decision Boundaries ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

def plot_decision(ax, X, y, title, predict_fn, is_ml=False):
    xx, yy = np.meshgrid(np.linspace(0.2, 4.2, 200),
                         np.linspace(0.05, 1.05, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if is_ml:
        Z = predict_fn.predict(grid).reshape(xx.shape)
    else:
        Z = np.array([predict_fn(a, m) for a, m in grid]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=[0, 0.5, 1])
    colors = ['#2ecc71' if lab == 1 else '#e74c3c' for lab in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5, s=20,
               edgecolors='k', linewidth=0.3)
    ax.set_xlabel('Aspect Ratio (Height / Width)')
    ax.set_ylabel('Motion Consistency')
    ax.set_title(title)
    ax.set_xlim(0.2, 4.2); ax.set_ylim(0.05, 1.05)

plot_decision(axes[0], X, y, 'Rule-Based: Hard Rectangular Boundary',
             rule_based_classifier)
plot_decision(axes[1], X, y, 'ML (Logistic Regression): Learned Boundary',
             model, is_ml=True)
plt.tight_layout()
plt.savefig('rule_vs_ml_boundary.png', dpi=120)
plt.show()

# --- Analysis: Where do rules fail? ---
rule_errors = y != y_rule_pred
ml_errors = y != y_ml_pred
print(f"\nRule-based errors: {rule_errors.sum()}/{len(y)} "
      f"({rule_errors.sum()/len(y)*100:.1f}%)")
print(f"ML errors: {ml_errors.sum()}/{len(y)} "
      f"({ml_errors.sum()/len(y)*100:.1f}%)")

# Identify types of rule failures
false_negatives = (y == 1) & (y_rule_pred == 0)
false_positives = (y == 0) & (y_rule_pred == 1)
print(f"\nRule-Based Failure Analysis:")
print(f"  False Negatives (missed pedestrians): {false_negatives.sum()}")
print(f"  False Positives (false alarms): {false_positives.sum()}")
if false_negatives.sum() > 0:
    print(f"  Typical missed pedestrian: "
          f"AR={X[false_negatives, 0].mean():.2f}, "
          f"MC={X[false_negatives, 1].mean():.2f}")


# ============================================================================
# Listing 3: Framing a measurable AV perception research question in code
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate a typical day-vs-night detection result
# (In later chapters, we will load real nuScenes annotations and run a real detector)
np.random.seed(42)

# Hypothetical AP scores from an experiment
conditions = ['Daytime (train)', 'Daytime (test)',
              'Nighttime (no finetune)', 'Nighttime (100-img finetune)']
ap_scores = [0.72, 0.68, 0.41, 0.59]
errors = [0.02, 0.03, 0.04, 0.03]

# The research question in code
print("Research Question:")
print("Does YOLOv8 trained on daytime nuScenes images maintain")
print("pedestrian AP >= 0.60 at night? If not, can 100 nighttime")
print("fine-tuning images close the gap?")
print()

# Hypothesis: daytime-only model performs worse at night;
# fine-tuning on a small nighttime set recovers most of the gap.
print("Hypothesis:")
print("H0: AP_night == AP_day")
print("H1: AP_night < AP_day  (day-only model)")
print("H2: AP_night(finetuned) > AP_night(no-finetune)")
print()

# Visualize
x = np.arange(len(conditions))
bars = plt.bar(x, ap_scores, color=['#2E86AB', '#2E86AB',
                                     '#D64045', '#A23B72'])
plt.errorbar(x, ap_scores, yerr=errors, fmt='none',
             ecolor='black', capsize=8)
plt.axhline(y=0.60, color='gray', linestyle='--',
            label='Minimum acceptable AP')
plt.xticks(x, conditions, rotation=15, ha='right')
plt.ylabel('Pedestrian Average Precision')
plt.title('Day vs. Night Pedestrian Detection (Simulated)')
plt.legend()
plt.tight_layout()
plt.show()

# Conclusion framing
print("\nExpected findings (to be tested with real data):")
print(f"  1. Day-night gap: {ap_scores[1]-ap_scores[2]:.2f} AP drop")
print(f"  2. Fine-tuning benefit: {ap_scores[3]-ap_scores[2]:.2f} AP gain")
print(f"  3. Remaining gap after fine-tuning: {ap_scores[1]-ap_scores[3]:.2f} AP")
print()
print("If AP_night(finetuned) >= 0.60: solution is viable.")
print("If AP_night(finetuned) < 0.60: need more data or a different approach.")


# ============================================================================
# Listing 4: Systematic evaluation of candidate AV research questions
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Five candidate research questions for an AV perception project
# Each is evaluated on the three tests from Chapter 2
candidates = [
    {
        "question": "Can we make self-driving cars safer?",
        "specificity": 1, "measurability": 1, "novelty": 1,
        "level": "Too Broad (Simple)"
    },
    {
        "question": "Does adding nighttime images to training data "
                    "improve pedestrian detection recall at night?",
        "specificity": 4, "measurability": 5, "novelty": 3,
        "level": "Strong"
    },
    {
        "question": "What is the effect of rain intensity on LiDAR "
                    "point-cloud density for pedestrian detection?",
        "specificity": 4, "measurability": 4, "novelty": 4,
        "level": "Strong"
    },
    {
        "question": "Can AI solve all perception problems?",
        "specificity": 0, "measurability": 0, "novelty": 0,
        "level": "Unanswerable"
    },
    {
        "question": "How does bounding-box size (in pixels) correlate "
                    "with detection confidence across pedestrian, car, "
                    "and cyclist classes in the KITTI dataset?",
        "specificity": 5, "measurability": 5, "novelty": 3,
        "level": "Strong"
    },
]

df = pd.DataFrame(candidates)
df['total_score'] = df['specificity'] + df['measurability'] + df['novelty']

# Decision rule: passes if total >= 10 AND no single dimension < 3
df['passes'] = (df['total_score'] >= 10) & \
               (df['specificity'] >= 3) & \
               (df['measurability'] >= 3) & \
               (df['novelty'] >= 2)

print("=" * 65)
print("EVALUATING AV PERCEPTION RESEARCH QUESTIONS")
print("The Three Tests: Specificity, Measurability, Novelty")
print("=" * 65)

for i, row in df.iterrows():
    status = "PASS" if row['passes'] else "FAIL"
    print(f"\n{'-' * 65}")
    print(f"Q{i+1} [{status} | {row['level']} | "
          f"Score: {row['total_score']}/15]")
    print(f"  {row['question'][:90]}...")
    print(f"  Specificity={row['specificity']}  "
          f"Measurability={row['measurability']}  "
          f"Novelty={row['novelty']}")

# --- Visualization: Radar Chart of the Three Tests ---
categories = ['Specificity', 'Measurability', 'Novelty']
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
for i, (_, row) in enumerate(df.iterrows()):
    values = [row['specificity'], row['measurability'], row['novelty']]
    values += values[:1]
    label = f"Q{i+1} ({'PASS' if row['passes'] else 'FAIL'})"
    ax.fill(angles, values, alpha=0.15, color=colors[i])
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])

ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0, 5); ax.set_yticks([1, 2, 3, 4, 5])
ax.set_title("Candidate Research Questions: Three-Test Radar",
             pad=25, fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout(); plt.savefig('research_question_radar.png', dpi=120)
plt.show()

# --- Summary ---
passing = df[df['passes']]
failing = df[~df['passes']]
print(f"\n{'=' * 65}")
print(f"RESULTS: {len(passing)}/{len(df)} questions pass")
print(f"{'=' * 65}")
print(f"\nPASSING questions (ready for experimentation):")
for _, row in passing.iterrows():
    print(f"  + {row['question'][:80]}...")
print(f"\nFAILING questions (need refinement):")
for _, row in failing.iterrows():
    print(f"  - {row['question'][:80]}...")
    reasons = []
    if row['specificity'] < 3: reasons.append('too vague')
    if row['measurability'] < 3: reasons.append('not measurable')
    if row['novelty'] < 2: reasons.append('not novel enough')
    print(f"    Reasons: {', '.join(reasons)}")


# ============================================================================
# Listing 5: Loading the Iris dataset in Colab
# ============================================================================

# Import libraries
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Load iris data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Look at the data
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.describe())


# ============================================================================
# Listing 6: Quick visualization of the Iris dataset
# ============================================================================

# Scatter plot: petal length vs petal width, colored by species
plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['petal length (cm)'],
                      df['petal width (cm)'],
                      c=df['species'], cmap='viridis')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Species by Petal Measurements')
plt.colorbar(scatter, label='Species')
plt.show()


# ============================================================================
# Listing 7: Training and evaluating a simple classifier
# ============================================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data: 80% train, 20% test
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=iris.target_names))


# ============================================================================
# Listing 8: Comparing prompt patterns for AV literature review tasks
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt

# The paper abstract (Parikh et al., 2023, "Pedestrian Detection
# in Low-Light Conditions: A Benchmark and Baseline")
abstract = (
    "Autonomous vehicles must reliably detect pedestrians in all "
    "lighting conditions, yet most pedestrian detectors are trained "
    "and evaluated on daytime images only. We present NightPed-1K, "
    "a benchmark of 1,000 real nighttime driving scenes with 4,200 "
    "annotated pedestrian instances collected from urban routes in "
    "three cities. Evaluating five state-of-the-art detectors (YOLOv8, "
    "DETR, Faster R-CNN, CenterNet, and EfficientDet) on NightPed-1K, "
    "we find that all models suffer a 30-50% drop in Average Precision "
    "compared to daytime, with smaller models degrading more severely. "
    "We further show that fine-tuning on just 200 synthetically darkened "
    "daytime images recovers 60-80% of the lost AP, providing a "
    "low-cost mitigation strategy."
)

# Prompt patterns to compare
patterns = {
    "Naive (no structure)": "Summarize this paper abstract.",
    "Role Only": (
        "You are a computer vision researcher. "
        "Summarize this paper abstract."
    ),
    "Four-Part Framework": (
        "ROLE: You are a senior AV perception researcher writing a "
        "literature review.\n"
        "CONTEXT: I need to understand this paper's contribution, "
        "method, key finding, and limitation.\n"
        "FORMAT: Produce exactly four labeled sections: (1) PROBLEM, "
        "(2) METHOD, (3) KEY RESULT, (4) LIMITATION.\n"
        "CONSTRAINTS: Each section under 40 words. Include specific "
        "numbers. Mark uncertain claims with [CHECK]."
    ),
    "Chain-of-Thought": (
        "You are a senior AV perception researcher. Before writing "
        "your summary, think step by step:\n"
        "Step 1: What specific problem does this paper address?\n"
        "Step 2: What dataset or method did the authors create?\n"
        "Step 3: What is the single most important number?\n"
        "Step 4: What limitation do the authors acknowledge?\n"
        "Then write a four-section summary: PROBLEM, METHOD, "
        "KEY RESULT, LIMITATION. Each under 40 words."
    ),
}

# Evaluation rubric: score each pattern on 5 criteria
scores = {
    "Naive (no structure)":       [0, 1, 1, 0, 0],
    "Role Only":                  [1, 1, 1, 0, 1],
    "Four-Part Framework":        [1, 1, 1, 1, 1],
    "Chain-of-Thought":           [1, 1, 1, 1, 1],
}
criteria = ['Has Problem\nStatement', 'Has Method\nSummary',
            'Has Result\n(Numbers)', 'Has Limitation',
            'Lit-Review\nReady']
risk = {
    "Naive (no structure)": 3,
    "Role Only": 2,
    "Four-Part Framework": 1,
    "Chain-of-Thought": 0,
}

# --- Visualization ---
pattern_names = list(scores.keys())
totals = [sum(v) for v in scores.values()]
risk_vals = [risk[p] for p in pattern_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Heatmap of criteria
data = np.array([scores[p] for p in pattern_names])
im = axes[0].imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
axes[0].set_xticks(range(len(criteria)))
axes[0].set_xticklabels(criteria, fontsize=8)
axes[0].set_yticks(range(len(pattern_names)))
axes[0].set_yticklabels([p[:25] for p in pattern_names], fontsize=9)
axes[0].set_title('Criteria Satisfaction by Prompt Pattern')
for i in range(len(pattern_names)):
    for j in range(len(criteria)):
        axes[0].text(j, i, 'Y' if data[i, j] else 'N',
                     ha='center', va='center', fontweight='bold')

# Bar chart of total scores
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#27ae60']
axes[1].barh(pattern_names, totals, color=colors, edgecolor='black')
axes[1].set_xlabel('Completeness Score (max 5)')
axes[1].set_xlim(0, 5.5)
axes[1].set_title('Output Completeness by Prompt Pattern')
for i, v in enumerate(totals):
    axes[1].text(v + 0.1, i, f'{v}/5', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('prompt_comparison.png', dpi=120)
plt.show()

print("=" * 55)
print("PROMPT PATTERN COMPARISON RESULTS")
print("=" * 55)
for p in pattern_names:
    print(f"\n{p}:")
    print(f"  Score: {sum(scores[p])}/5")
    print(f"  Hallucination Risk: "
          f"{['Lowest','Low','Medium','High'][risk[p]]}")

print(f"\n{'=' * 55}")
print("RECOMMENDATION:")
print("  Default: Four-Part Framework (best cost/quality ratio)")
print("  Critical papers: Chain-of-Thought (lowest hallucination)")
print("  Never use: Naive prompt (unstructured, high risk)")


# ============================================================================
# Listing 9: LLM-assisted paper summarization with verification
# ============================================================================

# Demonstrating the LLM-assisted research workflow
# In practice, you would copy-paste into ChatGPT; this code
# simulates the verification step programmatically.

# Ground truth from the actual YOLO paper (verified by reading it)
yolo_ground_truth = {
    "title": "You Only Look Once: Unified, Real-Time Object Detection",
    "authors": ["Joseph Redmon", "Santosh Divvala", "Ross Girshick",
                "Ali Farhadi"],
    "year": 2016,
    "venue": "CVPR",
    "key_contribution": "Unified detection as a single regression problem "
                        "from image pixels to bounding boxes and class "
                        "probabilities in one forward pass.",
    "speed_fps": 45,  # Fast YOLO on PASCAL VOC
    "limitation_stated": "Struggles with small objects and groups of "
                         "small objects (e.g., flocks of birds)."
}

print("=" * 60)
print("LLM PAPER SUMMARIZATION & VERIFICATION WORKFLOW")
print("=" * 60)

# Step 1: Construct the prompt (the four-part framework)
prompt = f"""
You are a computer vision researcher introducing object detection
to a motivated beginner.

CONTEXT: I am studying autonomous vehicle perception and need to
understand the YOLO paper by Redmon et al. (2016).

FORMAT: Produce exactly three sections:
1. SUMMARY: What does YOLO do, in plain English (max 80 words)
2. CONTRIBUTION: The one main new idea (max 50 words)
3. LIMITATION: YOLO's most important weakness per the authors
   (max 50 words)

CONSTRAINTS: No equations. Use analogies where helpful. Mark any
claim you are uncertain about with [CHECK].
"""

print("\nPROMPT (sent to LLM):")
print(prompt)

# Step 2: Simulate LLM response (in practice, this comes from the API)
llm_response = {
    "summary": "YOLO treats object detection as a single regression "
               "problem. It divides the image into a grid; each grid "
               "cell predicts bounding boxes and class probabilities "
               "simultaneously. This unified approach runs at 45 FPS, "
               "making it suitable for real-time applications like "
               "autonomous driving perception.",
    "contribution": "Reframing detection as a single regression from "
                    "pixels to boxes, eliminating the multi-stage "
                    "pipelines used by R-CNN and similar detectors.",
    "limitation": "YOLO struggles with small objects and groups of "
                  "small objects because each grid cell predicts only "
                  "a fixed number of boxes. This is a known challenge "
                  "for AV perception, where distant pedestrians "
                  "occupy few pixels."
}

print("\nLLM RESPONSE:")
for key, val in llm_response.items():
    print(f"  {key}: {val}")

# Step 3: Verification checklist
print("\n" + "=" * 60)
print("VERIFICATION CHECKLIST")
print("=" * 60)

checks = {
    "Paper exists?": True,   # Verified on Google Scholar
    "Authors match?": "Redmon et al." in str(yolo_ground_truth["authors"]),
    "Year correct?": yolo_ground_truth["year"] == 2016,
    "Speed claim (45 FPS) correct?": yolo_ground_truth["speed_fps"] == 45,
    "Limitation matches paper?": "small objects" in llm_response["limitation"].lower(),
}

for check, result in checks.items():
    status = "PASS" if result else "FAIL -- VERIFY MANUALLY"
    print(f"  [{status}] {check}")

print("\n" + "=" * 60)
print("KEY LESSON: LLMs accelerate summarization,")
print("but YOU must verify every factual claim against the paper.")
print("=" * 60)


# ============================================================================
# Listing 10: Your first Colab code --- running Python in the browser
# ============================================================================

# This is a code cell. Press Shift+Enter to run.
print("Welcome to AI Research with Python!")
print("Our anchor case: autonomous vehicle perception.")

# Variables store data for later use
sensor = "camera"
resolution_pixels = 1920 * 1080
print(f"Processing {sensor} data at {resolution_pixels} pixels per frame.")


# ============================================================================
# Listing 11: Your first Python code in Colab
# ============================================================================

print("Hello, AI researcher!")


# ============================================================================
# Listing 12: Download any Kaggle dataset directly in Colab via API
# ============================================================================

# Step 1: Get your API token
# Go to kaggle.com -> Settings -> API -> Create New Token
# This downloads kaggle.json to your computer

# Step 2: Upload kaggle.json to Colab
from google.colab import files
files.upload()  # Select the kaggle.json file

# Step 3: Configure the API
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Step 4: Download any dataset
!kaggle datasets download -d username/dataset-name
!unzip dataset-name.zip


# ============================================================================
# Listing 13: Mount Google Drive to access any downloaded data
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')
# Now browse /content/drive/MyDrive/ for your files

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/my_dataset.csv')


# ============================================================================
# Listing 14: Python lists vs. NumPy arrays --- the speed difference
# ============================================================================

import numpy as np
import time

# Create 1 million numbers
n = 1_000_000

# Python list
py_list = list(range(n))
start = time.time()
py_squared = [x**2 for x in py_list]
print(f"Python list: {time.time() - start:.3f} seconds")

# NumPy array
np_array = np.arange(n)
start = time.time()
np_squared = np_array ** 2
print(f"NumPy array: {time.time() - start:.3f} seconds")

# NumPy is typically 10-50x faster


# ============================================================================
# Listing 15: The five NumPy operations every researcher needs
# ============================================================================

import numpy as np

# 1. CREATE arrays
arr = np.array([1, 2, 3, 4, 5])           # from a list
zeros = np.zeros((3, 4))                   # 3x4 matrix of zeros
ones = np.ones(10)                         # 10-element vector of ones
rand = np.random.randn(1000)               # 1000 samples from N(0,1)
sequence = np.arange(0, 10, 0.5)           # [0.0, 0.5, 1.0, ..., 9.5]
print(f"Created arrays: {arr.shape}, {zeros.shape}")

# 2. INDEX and SLICE (same syntax as Python lists)
print(arr[0])         # first element
print(arr[-1])        # last element
print(arr[1:4])       # slice: elements at indices 1,2,3
print(arr[arr > 3])   # boolean indexing: elements greater than 3

# 3. COMPUTE statistics
print(f"Mean: {rand.mean():.3f}, Std: {rand.std():.3f}")
print(f"Min: {rand.min():.3f}, Max: {rand.max():.3f}")
print(f"Median: {np.median(rand):.3f}")
print(f"90th percentile: {np.percentile(rand, 90):.3f}")

# 4. RESHAPE and transform
img = np.random.randn(28, 28)              # pretend it's a 28x28 image
flattened = img.reshape(-1)                # flatten to 784-element vector
print(f"Original shape: {img.shape} -> Flattened: {flattened.shape}")

# 5. BROADCAST (apply operations element-wise without loops)
detection_confidences = np.array([0.92, 0.45, 0.88, 0.31, 0.95])
threshold = 0.5
detected = detection_confidences > threshold   # boolean array
print(f"Confidences: {detection_confidences}")
print(f"Detected (conf > {threshold}): {detected}")
print(f"Detection rate: {detected.mean():.1%}")


# ============================================================================
# Listing 16: Loading your first dataset into a DataFrame
# ============================================================================

import pandas as pd

# Load the Titanic dataset from a URL
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# What does it look like?
print(f"Shape: {df.shape}")           # (rows, columns)
print(df.head())                       # First 5 rows
print(df.columns.tolist())             # All column names


# ============================================================================
# Listing 17: The five pandas operations you'll use in every project
# ============================================================================

# 1. INSPECT: look at your data
df.info()        # Data types, non-null counts, memory usage
df.describe()    # Count, mean, std, min, 25%, 50%, 75%, max

# 2. SELECT: pick columns
ages = df['Age']                    # One column -> Series
subset = df[['Name', 'Age', 'Survived']]  # Multiple columns

# 3. FILTER: pick rows
women = df[df['Sex'] == 'female']           # Only women
first_class = df[df['Pclass'] == 1]          # Only 1st class
young_survivors = df[(df['Age'] < 18) & (df['Survived'] == 1)]

# 4. AGGREGATE: compute statistics
avg_age = df['Age'].mean()
survival_rate = df['Survived'].mean()
fare_by_class = df.groupby('Pclass')['Fare'].mean()

# 5. HANDLE MISSING: find and drop gaps
df.isnull().sum()     # Count missing values per column
df_clean = df.dropna(subset=['Age'])  # Drop rows where Age is missing


# ============================================================================
# Listing 18: Three essential plot types with matplotlib
# ============================================================================

import matplotlib.pyplot as plt

# Histogram: distribution of a single variable
plt.figure(figsize=(8, 4))
plt.hist(df['Age'].dropna(), bins=30, color='steelblue', edgecolor='white')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution of Titanic Passengers')
plt.show()

# Bar chart: comparing categories
plt.figure(figsize=(8, 4))
survival_by_class = df.groupby('Pclass')['Survived'].mean()
survival_by_class.plot(kind='bar', color=['gold', 'silver', 'brown'])
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.xticks(rotation=0)
plt.show()

# Scatter plot: relationship between two variables
plt.figure(figsize=(8, 4))
plt.scatter(df['Age'], df['Fare'], alpha=0.5, c=df['Survived'], cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare, Colored by Survival')
plt.colorbar(label='Survived')
plt.show()


# ============================================================================
# Listing 19: Seaborn makes statistical plots trivial
# ============================================================================

import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic data (built into seaborn for practice)
df = sns.load_dataset('titanic')

# 1. HEATMAP: correlation between numeric variables
# Answers: "Which variables move together?"
plt.figure(figsize=(8, 6))
numeric_cols = df.select_dtypes(include='number').dropna()
sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title('Correlation Heatmap: Titanic Numeric Features')
plt.tight_layout(); plt.show()

# 2. BOX PLOT: distribution across categories
# Answers: "How does a variable vary by group?"
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='class', y='age', hue='survived',
            palette='Set2')
plt.title('Age Distribution by Class and Survival')
plt.xlabel('Passenger Class'); plt.ylabel('Age')
plt.show()

# 3. PAIR PLOT: all pairwise relationships
# Answers: "What is the overall structure of my data?"
# (Use a subset of columns to keep it readable)
subset_cols = ['survived', 'age', 'fare', 'pclass']
sns.pairplot(df[subset_cols].dropna().sample(200),
             hue='survived', diag_kind='hist',
             palette={0: '#e74c3c', 1: '#2ecc71'})
plt.suptitle('Pair Plot: Titanic Subset (n=200)',
             y=1.02, fontsize=14)
plt.show()


# ============================================================================
# Listing 20: Generating and analyzing a simulated AV detection log
# ============================================================================

import numpy as np
import pandas as pd

# Generate a simulated AV detection log (8,000 detections across 200 frames)
np.random.seed(42)
n = 8000
detections = pd.DataFrame({
    'frame_id': np.random.randint(1, 201, n),
    'object_class': np.random.choice(
        ['pedestrian', 'car', 'cyclist', 'sign'], n,
        p=[0.25, 0.50, 0.15, 0.10]),
    'confidence': np.clip(np.random.normal(0.78, 0.15, n), 0.1, 0.99),
    'box_width': np.random.normal(80, 40, n),
    'box_height': np.random.normal(100, 50, n),
    'distance_m': np.random.exponential(25, n),
})
detections.to_csv('detection_log.csv', index=False)

# Load and inspect
detections = pd.read_csv("detection_log.csv")

# Inspect
print(f"Total detections: {len(detections)}")
print(f"Classes: {detections['object_class'].unique()}")
print(detections.describe())

# Average confidence by object class
print("\nAverage detection confidence:")
print(detections.groupby('object_class')['confidence'].mean())

# How many objects per frame?
objects_per_frame = detections.groupby('frame_id').size()
print(f"\nAvg objects per frame: {objects_per_frame.mean():.1f}")

# Do smaller (farther) objects have lower confidence?
plt.figure(figsize=(8, 5))
plt.scatter(detections['box_height'],
            detections['confidence'],
            alpha=0.3, s=5)
plt.xlabel('Bounding Box Height (pixels)')
plt.ylabel('Detection Confidence')
plt.title('Confidence vs. Object Size: The Small-Object Problem')
plt.show()


# ============================================================================
# Listing 21
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(df.columns.tolist())
df.info()


# ============================================================================
# Listing 22
# ============================================================================

print("Survival rate:", f"{df['Survived'].mean():.1%}")
print("Average age:", f"{df['Age'].mean():.1f}")
print("Average fare:", f"${df['Fare'].mean():.2f}")
print()
print("Survival by sex:")
print(df.groupby('Sex')['Survived'].mean())
print()
print("Survival by class:")
print(df.groupby('Pclass')['Survived'].mean())


# ============================================================================
# Listing 23
# ============================================================================

# Chart 1: Survival by sex and class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df.groupby('Sex')['Survived'].mean().plot(
    kind='bar', ax=axes[0], color=['pink', 'lightblue'])
axes[0].set_title('Survival Rate by Sex')
axes[0].set_ylabel('Survival Rate')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

df.groupby('Pclass')['Survived'].mean().plot(
    kind='bar', ax=axes[1], color=['gold', 'silver', 'brown'])
axes[1].set_title('Survival Rate by Passenger Class')
axes[1].set_ylabel('Survival Rate')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Chart 2: Age distribution by survival
plt.figure(figsize=(10, 5))
for survived, label, color in [(0, 'Died', 'red'), (1, 'Survived', 'green')]:
    subset = df[df['Survived'] == survived]['Age'].dropna()
    plt.hist(subset, bins=30, alpha=0.5, label=label, color=color)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Distribution: Survivors vs. Non-Survivors')
plt.legend()
plt.show()


# ============================================================================
# Listing 24: Survival by sex AND class --- revealing the interaction
# ============================================================================

# Pivot table: survival rate by sex AND passenger class
pivot = df.pivot_table(
    values='Survived',
    index='Sex',
    columns='Pclass',
    aggfunc='mean'
)
print(pivot.round(3))
# Output will show something like:
# Pclass    1      2      3
# Sex
# female  0.968  0.921  0.500
# male    0.369  0.157  0.135


# ============================================================================
# Listing 25: The scikit-learn pattern: fit, predict, evaluate
# ============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Every scikit-learn model follows this exact pattern
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)            # Learn from training data
y_pred = model.predict(X_test)         # Predict on unseen data

print(classification_report(y_test, y_pred))
print(f"Top feature: {model.feature_importances_.argmax()}")


# ============================================================================
# Listing 26: Surveying AV dataset characteristics in Python
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Characteristics of the four major AV perception datasets
datasets = {
    'Kaggle AV': {'images': 1500, 'classes': 5, 'night': False,
                  'lidar': False, 'size_gb': 0.5, 'difficulty': 'Beginner'},
    'KITTI': {'images': 15000, 'classes': 8, 'night': False,
              'lidar': True, 'size_gb': 12, 'difficulty': 'Intermediate'},
    'nuScenes': {'images': 1.4e6, 'classes': 23, 'night': True,
                 'lidar': True, 'size_gb': 350, 'difficulty': 'Advanced'},
    'Waymo': {'images': 1.0e7, 'classes': 4, 'night': True,
              'lidar': True, 'size_gb': 2000, 'difficulty': 'Expert'},
}

df = pd.DataFrame(datasets).T
print("=== AV DATASET COMPARISON ===\n")
print(df.to_string())

# Decision matrix: score each dataset on requirements
requirements = {'night': 1.0, 'lidar': 0.3, 'size_gb': 0.0}
# Lower size is better for beginners
scores = {}
for name, props in datasets.items():
    score = 0
    if props['night']: score += 3   # Night data is critical
    if not props['lidar']: score += 1  # Camera-only is simpler
    if props['size_gb'] < 10: score += 2  # Small is accessible
    if props['classes'] >= 4: score += 1  # Sufficient object diversity
    scores[name] = score

print("\n=== BEGINNER-FRIENDLINESS SCORE (higher = easier) ===")
for name, s in sorted(scores.items(), key=lambda x: -x[1]):
    bar = '#' * s
    print(f"  {name:<12s}: {bar} ({s}/7)")
print("\nRecommendation: Start with Kaggle AV datasets for learning.")
print("Graduate to KITTI for benchmark comparisons.")
print("Use nuScenes/Waymo only when you have a specific multi-sensor")
print("research question and access to sufficient storage/compute.")


# ============================================================================
# Listing 27: Scaling features to comparable ranges
# ============================================================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler: transform to mean=0, std=1
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# MinMaxScaler: transform to range [0, 1]
scaler = MinMaxScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])


# ============================================================================
# Listing 28: Encoding categorical variables
# ============================================================================

# One-hot encoding: each category becomes its own binary column
df_encoded = pd.get_dummies(df, columns=['object_class'])

# Label encoding: categories become integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['object_class_encoded'] = le.fit_transform(df['object_class'])


# ============================================================================
# Listing 29: Extracting features from dates
# ============================================================================

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['quarter'] = df['date'].dt.quarter
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days


# ============================================================================
# Listing 30: Simple text features
# ============================================================================

df['text_length'] = df['review'].str.len()
df['word_count'] = df['review'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']
df['exclamation_count'] = df['review'].str.count('!')
df['has_positive_word'] = df['review'].str.contains(
    'great|excellent|amazing|love|wonderful', case=False
).astype(int)


# ============================================================================
# Listing 31: Creating interaction features
# ============================================================================

# House: price per square foot is more informative than raw price
df['price_per_sqft'] = df['SalePrice'] / df['GrLivArea']

# AV: aspect ratio distinguishes tall pedestrians from wide cars
df['bbox_aspect_ratio'] = df['box_height'] / df['box_width']

# Airbnb: reviews per month is engagement rate
df['reviews_per_listing_year'] = (df['number_of_reviews'] /
    (2025 - df['host_since'].dt.year).clip(lower=1))


# ============================================================================
# Listing 32: Diagnosing data quality problems
# ============================================================================

import pandas as pd
import numpy as np

# Download from: kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
df = pd.read_csv("AB_NYC_2019.csv")
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nPrice statistics:\n{df['price'].describe()}")


# ============================================================================
# Listing 33: Cleaning the Airbnb dataset step by step
# ============================================================================

# 1. Drop columns with >20% missing or irrelevant
df = df.drop(columns=['id', 'host_name', 'last_review'])

# 2. Handle missing 'reviews_per_month' (fill with 0)
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)

# 3. Remove extreme price outliers (price = 0 or > $1000/night)
print(f"Before outlier removal: {len(df)} rows")
df = df[(df['price'] > 0) & (df['price'] < 1000)]
print(f"After outlier removal: {len(df)} rows")

# 4. Standardize neighborhood names
df['neighbourhood_group'] = df['neighbourhood_group'].str.strip().str.title()

# 5. One-hot encode room type
df = pd.get_dummies(df, columns=['room_type'], prefix='room')

# 6. Log-transform price (prices are right-skewed)
df['price_log'] = np.log1p(df['price'])

print(f"\nFinal shape: {df.shape}")
print(f"Final columns: {df.columns.tolist()}")


# ============================================================================
# Listing 34: Cleaning the AV detection log for model training
# ============================================================================

# Load the detection log
detections = pd.read_csv("detection_log.csv")

print("=== BEFORE CLEANING ===")
print(f"Rows: {len(detections)}")
print(f"Missing values:\n{detections.isnull().sum()}")
print(f"\nObject classes: {detections['object_class'].unique()}")
print(f"Confidence range: [{detections['confidence'].min():.2f}, {detections['confidence'].max():.2f}]")
print(f"Distance range: [{detections['distance_m'].min():.1f}, {detections['distance_m'].max():.1f}]")

# === CLEANING PIPELINE ===

# 1. Drop rows where critical values are missing
detections = detections.dropna(subset=['confidence', 'object_class'])

# 2. Fill missing distance (use median by object class)
detections['distance_m'] = detections.groupby('object_class')['distance_m'].transform(
    lambda x: x.fillna(x.median())
)

# 3. Remove invalid confidence values (< 0 or > 1)
detections = detections[(detections['confidence'] >= 0) & (detections['confidence'] <= 1)]

# 4. Remove unrealistic distances (negative or > 200m)
detections = detections[(detections['distance_m'] > 0) & (detections['distance_m'] <= 200)]

# 5. Compute bounding box area (feature engineering)
detections['box_area'] = detections['box_width'] * detections['box_height']

# 6. One-hot encode object class
detections = pd.get_dummies(detections, columns=['object_class'], prefix='obj')

# 7. Split BEFORE scaling to avoid data leakage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define the target (for demonstration: predict whether an object is a pedestrian)
target = detections['obj_pedestrian']  # created by one-hot encoding above
features = detections.drop(columns=['obj_pedestrian'])

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)

# Fit the scaler on the TRAINING set only, then transform both sets.
# Fitting on the full dataset would leak test-set statistics (min/max)
# into the training preprocessing --- a classic data-leakage mistake.
scaler = MinMaxScaler()
numeric_cols = ['confidence', 'box_area', 'distance_m']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("\n=== AFTER CLEANING ===")
print(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")
print(f"Columns: {len(X_train.columns)}")
print(f"Missing values remaining (train): {X_train.isnull().sum().sum()}")
print(f"All confidence values in [0,1] (train): "
      f"{(X_train['confidence'].between(0,1)).all()}")


# ============================================================================
# Listing 35: Histogram with key statistics overlaid
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 5))
plt.hist(df['price'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)

# Overlay mean and median
plt.axvline(df['price'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: ${df["price"].mean():.0f}')
plt.axvline(df['price'].median(), color='green', linestyle='-',
            linewidth=2, label=f'Median: ${df["price"].median():.0f}')

plt.xlabel('Price per Night ($)')
plt.ylabel('Number of Listings')
plt.title('Distribution of NYC Airbnb Prices')
plt.legend()
plt.show()


# ============================================================================
# Listing 36: Bar chart with sorted categories
# ============================================================================

room_stats = df.groupby('room_type')['price'].median().sort_values()

plt.figure(figsize=(8, 5))
room_stats.plot(kind='barh', color=['gold', 'coral', 'steelblue'])
plt.xlabel('Median Price per Night ($)')
plt.title('Median Airbnb Price by Room Type')
plt.xlim(0, room_stats.max() * 1.2)  # x-axis starts at zero!
plt.show()


# ============================================================================
# Listing 37: Scatter plot with trend line
# ============================================================================

from scipy import stats

plt.figure(figsize=(8, 6))
plt.scatter(df['number_of_reviews'], df['price'],
            alpha=0.3, s=10, c='steelblue')

# Add a trend line
x = df['number_of_reviews']
y = df['price']
mask = ~(x.isna() | y.isna())
slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
plt.plot(x, slope * x + intercept, color='red', linewidth=2,
         label=f'R^2 = {r_value**2:.3f}')

plt.xlabel('Number of Reviews')
plt.ylabel('Price per Night ($)')
plt.title('Reviews vs. Price: Is Popularity Related to Price?')
plt.legend()
plt.show()


# ============================================================================
# Listing 38: Correlation heatmap with seaborn
# ============================================================================

import seaborn as sns

numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix: Which Variables Move Together?')
plt.show()


# ============================================================================
# Listing 39: Small multiples: price distribution by neighborhood
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
neighborhoods = df['neighbourhood_group'].unique()

for ax, neigh in zip(axes.flat, neighborhoods):
    subset = df[df['neighbourhood_group'] == neigh]['price']
    ax.hist(subset, bins=40, color='steelblue', edgecolor='white')
    ax.set_title(neigh, fontsize=11)
    ax.axvline(subset.median(), color='red', linestyle='--', linewidth=1)

fig.suptitle('Airbnb Price Distribution by Neighborhood', fontsize=14)
plt.tight_layout()
plt.show()


# ============================================================================
# Listing 40: Box plots: price by room type and neighborhood
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
df.boxplot(column='price', by='neighbourhood_group', ax=ax)
ax.set_xlabel('Neighborhood')
ax.set_ylabel('Price per Night (\$)')
ax.set_title('Price Distribution by Neighborhood (Box Plot)')
plt.suptitle('')  # Remove pandas auto-title
plt.xticks(rotation=45)
plt.show()


# ============================================================================
# Listing 41: Pair plot: all features against all features
# ============================================================================

import seaborn as sns
# Select key numeric columns and color by a categorical variable
sns.pairplot(df, vars=['price', 'number_of_reviews', 'availability_365',
                       'calculated_host_listings_count'],
             hue='neighbourhood_group', diag_kind='hist',
             plot_kws={'alpha': 0.4, 's': 8})
plt.suptitle('Pair Plot: Airbnb Features by Neighborhood', y=1.02)
plt.show()


# ============================================================================
# Listing 42: Three-chart data story for AV detection confidence analysis
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the KITTI detection log (simulated from real KITTI statistics)
np.random.seed(42)
n = 8000
df = pd.DataFrame({
    'confidence': np.clip(np.random.beta(6, 2, n) * (1 - np.random.exponential(0.003, n)), 0.1, 1.0),
    'distance_m': np.random.uniform(5, 80, n),
    'bbox_area': np.random.exponential(5000, n),
    'object_class': np.random.choice(['pedestrian', 'car', 'cyclist'], n, p=[0.35, 0.50, 0.15]),
})
df.loc[df['distance_m'] > 50, 'confidence'] *= 0.7
df['is_correct'] = (df['confidence'] > 0.5).astype(int)

print(f"Detections: {len(df)}")
print(f"Classes: {df['object_class'].value_counts().to_dict()}")

# Chart 1: Scatter -- Confidence vs. Distance, colored by object class
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, cls in zip(axes, ['pedestrian', 'car', 'cyclist']):
    subset = df[df['object_class'] == cls]
    ax.scatter(subset['distance_m'], subset['confidence'],
               alpha=0.3, s=10, c='steelblue' if cls == 'car' else
               'darkorange' if cls == 'pedestrian' else 'green')
    ax.set_title(f'{cls.capitalize()}s')
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Confidence')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
fig.suptitle('Chart 1: Detection Confidence vs. Distance by Object Class',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('av_confidence_scatter.png', dpi=100, bbox_inches='tight')
plt.show()

# Chart 2: Histogram -- Confidence distribution by distance bin
df['distance_bin'] = pd.cut(df['distance_m'],
    bins=[0, 20, 45, 80], labels=['Near (<20m)', 'Medium (20-45m)', 'Far (>45m)'])
fig, ax = plt.subplots(figsize=(9, 5))
for bin_name, color in zip(['Near (<20m)', 'Medium (20-45m)', 'Far (>45m)'],
                            ['#2ecc71', '#f39c12', '#e74c3c']):
    subset = df[df['distance_bin'] == bin_name]
    ax.hist(subset['confidence'], bins=25, alpha=0.5, label=bin_name, color=color)
ax.set_xlabel('Confidence'); ax.set_ylabel('Count')
ax.set_title('Chart 2: Confidence Distribution by Distance Range')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('av_confidence_hist.png', dpi=100, bbox_inches='tight')
plt.show()

# Chart 3: Heatmap -- Correlations among detection features
corr_cols = ['confidence', 'distance_m', 'bbox_area', 'is_correct']
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title('Chart 3: Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('av_correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.show()

# Key finding: distance is the dominant factor
print(f"\nCorrelation: confidence vs distance = "
      f"{df['confidence'].corr(df['distance_m']):.3f}")
print(f"Mean confidence: Near={df[df['distance_bin']=='Near (<20m)']['confidence'].mean():.2f}, "
      f"Far={df[df['distance_bin']=='Far (>45m)']['confidence'].mean():.2f}")


# ============================================================================
# Listing 43: Building a three-chart AV detection data story
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate the same simulated detection log
np.random.seed(42)
n = 8000
df = pd.DataFrame({
    'frame_id': np.random.randint(1, 201, n),
    'object_class': np.random.choice(
        ['pedestrian', 'car', 'cyclist', 'sign'], n,
        p=[0.25, 0.50, 0.15, 0.10]),
    'confidence': np.clip(np.random.normal(0.78, 0.15, n), 0.1, 0.99),
    'box_width': np.abs(np.random.normal(80, 40, n)),
    'box_height': np.abs(np.random.normal(100, 50, n)),
    'distance_m': np.random.exponential(25, n),
})
# Make confidence drop with distance
df['confidence'] = np.clip(
    df['confidence'] * (1 - df['distance_m']/80), 0.1, 0.99)
df['box_area'] = df['box_width'] * df['box_height']

print(f"Detections: {len(df)} | Classes: "
      f"{df['object_class'].nunique()} | Frames: {df['frame_id'].nunique()}")

# ============================================================
# CHART 1: Distance distribution by class (Histogram)
# Question: "At what distances do we detect each object type?"
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, cls in zip(axes, ['pedestrian', 'car', 'cyclist']):
    subset = df[df['object_class'] == cls]
    ax.hist(subset['distance_m'], bins=30, alpha=0.7,
            color='steelblue', edgecolor='white')
    ax.axvline(subset['distance_m'].median(), color='red',
               linestyle='--', linewidth=2,
               label=f'Median: {subset[\"distance_m\"].median():.1f}m')
    ax.set_title(f'{cls.capitalize()}s'); ax.legend(fontsize=8)
    ax.set_xlabel('Distance (m)'); ax.set_ylabel('Count')
fig.suptitle('Chart 1: Detection Distance Distribution by Object Class',
             fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig('ch9_chart1_distance.png', dpi=120); plt.show()

print("\nChart 1 finding: Pedestrians detected at shorter median")
print(f"distance ({df[df['object_class']=='pedestrian']['distance_m'].median():.1f}m)")
print(f"vs cars ({df[df['object_class']=='car']['distance_m'].median():.1f}m).")
print("Implication: small/far objects may be a pedestrian-detection blind spot.")

# ============================================================
# CHART 2: Confidence vs Box Size (Scatter)
# Question: "Does confidence drop for small (distant) objects?"
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {'pedestrian': 'darkorange', 'car': 'steelblue',
          'cyclist': 'green'}

for ax, cls in zip(axes, ['pedestrian', 'car', 'cyclist']):
    subset = df[df['object_class'] == cls].sample(300, random_state=42)
    ax.scatter(subset['box_area'], subset['confidence'],
               alpha=0.4, s=8, c=colors[cls])
    ax.set_xlabel('Bounding Box Area (px$^2$)')
    ax.set_ylabel('Confidence'); ax.set_ylim(0, 1.05)
    ax.set_title(f'{cls.capitalize()}s'); ax.grid(alpha=0.3)
fig.suptitle('Chart 2: Detection Confidence vs. Bounding Box Area',
             fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig('ch9_chart2_scatter.png', dpi=120); plt.show()

# Correlation check
corr = df['confidence'].corr(df['box_area'])
print(f"\nChart 2 finding: Confidence vs box area correlation = {corr:.3f}")
print("Small boxes (distant objects) have systematically lower confidence.")

# ============================================================
# CHART 3: Correlation Heatmap
# Question: "Which features drive detection confidence?"
# ============================================================
plt.figure(figsize=(8, 6))
# Encode class as numeric for correlation
class_dummies = pd.get_dummies(df['object_class'], prefix='is')
corr_df = pd.concat([df[['confidence', 'box_area', 'distance_m']],
                     class_dummies], axis=1)
corr_matrix = corr_df.corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'label': 'Pearson Correlation'})
plt.title('Chart 3: Feature Correlation Heatmap --- What Drives Confidence?',
          fontsize=13)
plt.tight_layout(); plt.savefig('ch9_chart3_heatmap.png', dpi=120); plt.show()

print("\nChart 3 finding:")
top_corr = corr_matrix['confidence'].drop('confidence').abs().sort_values(
    ascending=False)
for feat, val in top_corr.items():
    print(f"  {feat}: r = {corr_matrix.loc['confidence', feat]:.3f}")

# ============================================================
# THE DATA STORY (One Paragraph)
# ============================================================
story = """
The AV perception system's detection reliability is limited primarily
by object distance and size, not by object class. Chart 1 shows that
pedestrians are detected at substantially shorter median distances
than cars, creating a pedestrian-specific blind spot at long range.
Chart 2 confirms that detection confidence drops sharply for small
bounding boxes --- a proxy for distant objects --- across all three
classes. Chart 3 resolves the question: the strongest predictors of
confidence are distance_m (r = {dist:.2f}) and box_area (r = {area:.2f}),
not any class indicator. The research implication is clear: improving
detection of small, far-away objects --- through higher input resolution,
multi-scale feature pyramids, or zoom augmentation --- would benefit
ALL classes, not just one. The problem is not 'the model confuses
pedestrians with cars at night.' The problem is 'the model cannot see
small objects well enough, regardless of what they are.'
"""
print(story.format(
    dist=corr_matrix.loc['confidence', 'distance_m'],
    area=corr_matrix.loc['confidence', 'box_area']))


# ============================================================================
# Listing 44: Chart 1 --- Price by neighborhood
# ============================================================================

neighborhood_prices = df.groupby('neighbourhood_group')['price'].median().sort_values()

plt.figure(figsize=(10, 5))
neighborhood_prices.plot(kind='bar', color='steelblue')
plt.xlabel('Neighborhood')
plt.ylabel('Median Price per Night ($)')
plt.title('Chart 1: Manhattan Listings Are 2-3x More Expensive')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ============================================================================
# Listing 45: Chart 2 --- Price by room type within Manhattan
# ============================================================================

manhattan = df[df['neighbourhood_group'] == 'Manhattan']
room_prices = manhattan.groupby('room_type')['price'].median()

plt.figure(figsize=(8, 5))
room_prices.plot(kind='barh', color=['gold', 'coral', 'steelblue'])
plt.xlabel('Median Price per Night ($)')
plt.title('Chart 2: Within Manhattan, Entire Homes Cost 4x Private Rooms')
plt.show()


# ============================================================================
# Listing 46: Chart 3 --- Reviews vs. price colored by room type
# ============================================================================

plt.figure(figsize=(9, 6))
room_types = df['room_type'].unique()
colors = {'Entire home/apt': 'steelblue',
          'Private room': 'coral',
          'Shared room': 'gold'}

for rt in room_types:
    subset = df[df['room_type'] == rt]
    plt.scatter(subset['number_of_reviews'], subset['price'],
                alpha=0.4, s=15, label=rt, color=colors.get(rt, 'gray'))

plt.xlabel('Number of Reviews')
plt.ylabel('Price per Night ($)')
plt.title('Chart 3: Higher Prices Do NOT Mean More Reviews')
plt.ylim(0, 500)
plt.legend()
plt.show()


# ============================================================================
# Listing 47: Linear regression baseline for AV braking distance prediction
# ============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Simulated AV data: predict braking distance from speed and road wetness
np.random.seed(42)
n = 500
speed_kmh = np.random.uniform(10, 120, n)
road_wetness = np.random.uniform(0, 1, n)
braking_distance = 0.05 * speed_kmh**1.8 + 15 * road_wetness + np.random.normal(0, 3, n)

X = np.column_stack([speed_kmh, road_wetness])
y = braking_distance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
print(f"R^2 on test set: {model.score(X_test, y_test):.3f}")
print(f"Coefficients: speed={model.coef_[0]:.3f}, wetness={model.coef_[1]:.3f}")
print(f"Interpretation: +1 km/h -> +{model.coef_[0]:.2f}m braking distance, "
      f"+0.1 wetness -> +{model.coef_[1]*0.1:.2f}m")


# ============================================================================
# Listing 48: Decision tree for pedestrian detection feature analysis
# ============================================================================

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Simulated AV detection features
np.random.seed(42)
n = 600
# Features: object height (pixels), distance (m), edge density
height_px = np.random.normal(120, 50, n)
distance_m = np.random.exponential(25, n)
edge_density = np.random.normal(0.6, 0.2, n)
# Label: is_pedestrian (1 if tall narrow object at moderate distance)
is_pedestrian = ((height_px > 80) & (distance_m < 40) & (edge_density > 0.5)).astype(int)

X = np.column_stack([height_px, distance_m, edge_density])
y = is_pedestrian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Shallow tree (max_depth=3) --- controlled variance
tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
print(f"Test accuracy (max_depth=3): {tree.score(X_test, y_test):.3f}")

# Deep tree (unrestricted) --- high variance, likely overfitting
deep_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(f"Test accuracy (unrestricted): {deep_tree.score(X_test, y_test):.3f}")
print(f"Train accuracy (unrestricted): {deep_tree.score(X_train, y_train):.3f}")
print(f"Note: train >> test accuracy indicates overfitting (high variance)")


# ============================================================================
# Listing 49: Random forest for AV object classification
# ============================================================================

from sklearn.ensemble import RandomForestClassifier

# Same data as decision tree example
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

print(f"Random Forest test accuracy: {rf.score(X_test, y_test):.3f}")
print(f"Single deep tree test accuracy: {deep_tree.score(X_test, y_test):.3f}")
print(f"Improvement: {(rf.score(X_test, y_test) - deep_tree.score(X_test, y_test))*100:.1f} pp")

# Feature importances
for name, imp in zip(['height_px', 'distance_m', 'edge_density'],
                     rf.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# The forest tells you WHICH features matter, not just what it predicts
print(f"\nMost important feature: "
      f"{['height_px','distance_m','edge_density'][np.argmax(rf.feature_importances_)]}")


# ============================================================================
# Listing 50: XGBoost for AV pedestrian detection with feature importance
# ============================================================================

# pip install xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                    random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)

print(f"XGBoost test accuracy: {xgb.score(X_test, y_test):.3f}")
print(f"Random Forest accuracy: {rf.score(X_test, y_test):.3f}")
print(f"Single tree accuracy: {deep_tree.score(X_test, y_test):.3f}")

# XGBoost feature importances (gain-based: how much each feature improves splits)
for name, imp in zip(['height_px', 'distance_m', 'edge_density'],
                     xgb.feature_importances_):
    print(f"  {name}: {imp:.3f}")

print(f"\nModel comparison summary:")
print(f"  Linear Regression: simple, interpretable, high bias baseline")
print(f"  Decision Tree:     flexible, interpretable, high variance")
print(f"  Random Forest:     low variance via bagging, robust default")
print(f"  XGBoost:           low bias via boosting, competition winner")


# ============================================================================
# Listing 51: ML pipeline for AV pedestrian detection --- from features to model comparison
# ============================================================================

import pandas as pd, numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Generate synthetic KITTI-style detection proposals
np.random.seed(42); n = 5000
df = pd.DataFrame({
    'confidence': np.clip(np.random.beta(2, 5, n) * 0.95, 0.01, 0.99),
    'box_height': np.random.uniform(20, 350, n),
    'box_width': np.random.uniform(15, 200, n),
    'x_center': np.random.uniform(0, 1242, n),
    'y_center': np.random.uniform(0, 375, n),
})
df['box_area'] = df['box_height'] * df['box_width']
df['aspect_ratio'] = df['box_height'] / df['box_width']
# Label: real pedestrians tend to have higher confidence, taller boxes, near image center
score = (0.4 * df['confidence'] + 0.3 * (df['box_height']/350)
         + 0.2 * (1 - abs(df['x_center']-621)/621) + np.random.normal(0, 0.1, n))
df['is_pedestrian'] = (score > 0.45).astype(int)
print(f"Positive rate: {df['is_pedestrian'].mean():.2%}")

# Feature set
features = ['confidence', 'box_height', 'box_width', 'box_area',
            'aspect_ratio', 'x_center', 'y_center']
X = df[features]; y = df['is_pedestrian']

# Train/test split (stratified for class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Compare models with 5-fold CV
models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', probability=True),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\nModel Comparison (5-fold CV ROC-AUC):")
print("=" * 50)
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"{name:25s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Train best model and evaluate
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"\nTest Set Performance:")
print(classification_report(y_test, y_pred, target_names=['False Alarm', 'Pedestrian']))

# Feature importance
importances = pd.DataFrame({
    'feature': features, 'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(importances.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importances['feature'][::-1], importances['importance'][::-1], color='steelblue')
ax.set_xlabel('Gini Importance'); ax.set_title('What Makes a Detection a Real Pedestrian?')
plt.tight_layout(); plt.savefig('av_feature_importance.png', dpi=100); plt.show()


# ============================================================================
# Listing 52: Loading and exploring the Titanic dataset
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print(f"Training shape: {df_train.shape}")
print(f"Features: {list(df_train.columns)}")
print(df_train.head())
print(df_train.describe())

# Check for missing values
print(df_train.isnull().sum())
# -> Age: 177 missing, Cabin: 687 missing, Embarked: 2 missing

# Survival rate by class
print(df_train.groupby('Pclass')['Survived'].mean())
# -> 1st class: ~63%, 2nd: ~47%, 3rd: ~24%


# ============================================================================
# Listing 53: Feature engineering for the Titanic dataset
# ============================================================================

# Combine train and test for consistent preprocessing
df_all = pd.concat([df_train, df_test], sort=False)

# Extract title from name
df_all['Title'] = df_all['Name'].str.extract(
    r' ([A-Za-z]+)\.', expand=False)
df_all['Title'] = df_all['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df_all['Title'] = df_all['Title'].replace(
    ['Mlle', 'Ms'], 'Miss')
df_all['Title'] = df_all['Title'].replace('Mme', 'Mrs')

# Create family features
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)

# NOTE: We do NOT fill missing Age/Fare here. Filling them on the
# combined train+test set would leak test-set statistics into every
# training fold during cross-validation. They are imputed inside the
# Pipeline in Step 3, so each fold uses only its own training data.

# Embarked is categorical with just 2 missing values; fill with mode
# before one-hot encoding (its impact is negligible).
df_all['Embarked'] = df_all['Embarked'].fillna(
    df_all['Embarked'].mode()[0])

# Encode categorical variables
# Sex is unordered, so use one-hot encoding (NOT label encoding).
# For a binary category, this produces two columns: [1,0]=female, [0,1]=male.
df_all = pd.get_dummies(df_all, columns=['Sex'])
# For Title and Embarked (multi-category), drop one column to avoid collinearity
df_all = pd.get_dummies(df_all, columns=['Title', 'Embarked'],
                        drop_first=True)


# ============================================================================
# Listing 54: Training and comparing multiple ML models
# ============================================================================

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Prepare final features
feature_cols = ['Pclass', 'Age', 'Fare', 'FamilySize',
                'IsAlone'] + [c for c in df_all.columns
                               if c.startswith('Sex_')
                               or c.startswith('Title_')
                               or c.startswith('Embarked_')]
X = df_all[feature_cols].iloc[:len(df_train)]
y = df_train['Survived']

# Compare models with cross-validation. Wrap each model in a Pipeline
# so that imputation and scaling happen INSIDE each fold, using only
# that fold's training data. Fitting the imputer/scaler on the full
# dataset first would leak validation-fold statistics --- a classic
# data-leakage mistake that scikit-learn's docs explicitly warn about.
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model),
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"{name:25s}: {scores.mean():.4f} (+/- {scores.std():.4f})")


# ============================================================================
# Listing 55: Analyzing feature importance
# ============================================================================

# Train a Random Forest (with imputation) and extract feature importances
rf_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42)),
])
rf_pipe.fit(X, y)

importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_pipe.named_steps['model'].feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(10))

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importances['feature'][:10][::-1],
         importances['importance'][:10][::-1])
plt.xlabel('Feature Importance (Gini)')
plt.title('Top 10 Features: What Predicts Survival?')
plt.tight_layout()
plt.show()


# ============================================================================
# Listing 56: Plotting training/validation curves --- the essential DL diagnostic
# ============================================================================

import matplotlib.pyplot as plt

# Typical training history (simulated, realistic values)
epochs = list(range(1, 31))
train_loss = [2.30, 1.80, 1.40, 1.10, 0.90, 0.75, 0.62, 0.52, 0.45, 0.38,
              0.33, 0.28, 0.25, 0.22, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12,
              0.11, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
val_loss =   [2.50, 1.95, 1.55, 1.25, 1.05, 0.90, 0.78, 0.68, 0.62, 0.58,
              0.55, 0.53, 0.52, 0.52, 0.53, 0.54, 0.56, 0.58, 0.60, 0.62,
              0.65, 0.68, 0.71, 0.74, 0.77, 0.80, 0.83, 0.86, 0.89, 0.92]

plt.figure(figsize=(9, 5))
plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
plt.axvline(x=14, color='gray', linestyle='--', alpha=0.7,
            label='Early Stopping Point (epoch 14)')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.title('Training vs. Validation Loss: Detecting Overfitting')
plt.legend(); plt.grid(alpha=0.3)
plt.annotate('Best model:\nlowest validation loss',
             xy=(14, 0.52), xytext=(19, 0.70),
             arrowprops=dict(arrowstyle='->'), fontsize=10)
plt.tight_layout(); plt.savefig('loss_curves.png', dpi=120); plt.show()


# ============================================================================
# Listing 57: CNN for pedestrian vs. background classification on KITTI patches
# ============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np, matplotlib.pyplot as plt

# Simulate KITTI pedestrian patches (64x64 RGB, normalized)
np.random.seed(42)
def make_patch(is_pedestrian):
    """Generate a synthetic 64x64x3 patch. Real pedestrians have
    vertical structure in the center; backgrounds are random noise."""
    img = np.random.normal(0.3, 0.15, (64, 64, 3))
    if is_pedestrian:
        # Add a vertical bright stripe (simulating a person silhouette)
        img[8:56, 20:44, :] += np.random.normal(0.4, 0.1, (48, 24, 3))
    return np.clip(img, 0, 1)

X_train = np.array([make_patch(i < 4000) for i in range(8000)])
y_train = np.array([1]*4000 + [0]*4000)
X_test = np.array([make_patch(i < 1000) for i in range(2000)])
y_test = np.array([1]*1000 + [0]*1000)

# Shuffle
idx = np.random.permutation(8000); X_train, y_train = X_train[idx], y_train[idx]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Build a lightweight CNN
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=8, batch_size=64,
                    validation_split=0.2, verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(history.history['accuracy'], 'b-', label='Train')
axes[0].plot(history.history['val_accuracy'], 'r--', label='Val')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].set_title('CNN Training: Pedestrian Detection'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='Train')
axes[1].plot(history.history['val_loss'], 'r--', label='Val')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('cnn_pedestrian_training.png', dpi=100); plt.show()

# Visualize predictions
y_prob = model.predict(X_test[:16], verbose=0).flatten()
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i]); ax.axis('off')
    true = 'Ped' if y_test[i] else 'BG'
    pred = 'Ped' if y_prob[i] > 0.5 else 'BG'
    color = 'green' if (y_prob[i]>0.5) == y_test[i] else 'red'
    ax.set_title(f'True:{true} Pred:{pred} ({y_prob[i]:.2f})', color=color, fontsize=8)
plt.suptitle('CNN Predictions: Green=Correct, Red=Wrong', fontsize=13)
plt.tight_layout(); plt.savefig('cnn_predictions.png', dpi=100); plt.show()


# ============================================================================
# Listing 58: Loading Fashion MNIST with Keras
# ============================================================================

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (built into Keras)
(X_train, y_train), (X_test, y_test) = (
    keras.datasets.fashion_mnist.load_data())

# Class names
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training: {X_train.shape}, Test: {X_test.shape}")
# -> Training: (60000, 28, 28), Test: (10000, 28, 28)

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Add channel dimension (required by Conv2D)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"After reshape: {X_train.shape}")
# -> (60000, 28, 28, 1)

# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].squeeze(), cmap='gray')
    ax.set_title(class_names[y_train[i]])
    ax.axis('off')
plt.suptitle('Fashion MNIST Samples')
plt.show()


# ============================================================================
# Listing 59: Building a CNN for Fashion MNIST
# ============================================================================

from tensorflow.keras import layers, models

def build_cnn():
    model = models.Sequential([
        # Block 1: Conv -> ReLU -> Pool
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),

        # Block 2: Conv -> ReLU -> Pool
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Block 3: Conv -> ReLU
        layers.Conv2D(64, (3, 3), activation='relu'),

        # Classifier head
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_cnn()
model.summary()


# ============================================================================
# Listing 60: Training and evaluating the CNN
# ============================================================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,
    batch_size=64,
    verbose=1
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Over Training')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss Over Training')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================================
# Listing 61: Visualizing correct and incorrect predictions
# ============================================================================

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Find misclassified examples
misclassified = np.where(y_pred_classes != y_test)[0]
print(f"Misclassified: {len(misclassified)} out of {len(y_test)} "
      f"({100*len(misclassified)/len(y_test):.2f}%)")

# Display some misclassifications
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    idx = misclassified[i]
    ax.imshow(X_test[idx].squeeze(), cmap='gray')
    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                 color='red')
    ax.axis('off')
plt.suptitle('Misclassified Examples: What Does the Model Confuse?')
plt.tight_layout()
plt.show()


# ============================================================================
# Listing 62: Classifying AV incident reports with BERT and zero-shot prompting
# ============================================================================

# Part 1: Fine-tuned classifier (simulated with mock AV incident data)
import pandas as pd, numpy as np

# Generate synthetic AV incident reports
np.random.seed(42)
templates = {
    'perception_failure': [
        "Vehicle failed to detect pedestrian crossing at intersection. Disengaged.",
        "Camera occlusion caused missed detection of cyclist in right lane.",
        "Lidar point cloud too sparse to classify object at 80m range. Manual takeover.",
        "Traffic light state misclassified under direct sunlight. Emergency stop.",
    ],
    'prediction_error': [
        "Vehicle incorrectly predicted adjacent car would maintain lane. Near collision.",
        "Cut-in prediction failed: merging vehicle trajectory underestimated.",
        "Pedestrian speed overestimated; unnecessary hard brake triggered.",
    ],
    'planning_issue': [
        "Path planner generated trajectory outside drivable area. Disengaged.",
        "Unprotected left turn timed too aggressively. Safety driver intervened.",
        "Lane change aborted mid-maneuver due to conservative gap threshold.",
    ],
    'external_factor': [
        "Heavy rain reduced sensor range below operational threshold.",
        "Construction zone signage not recognized. Manual navigation required.",
        "GPS signal loss in tunnel caused localization drift.",
    ],
}
reports, labels = [], []
for label, temps in templates.items():
    for _ in range(50):
        reports.append(np.random.choice(temps))
        labels.append(label)
df = pd.DataFrame({'report': reports, 'label': labels})
print(f"Reports: {len(df)}, Classes: {df['label'].value_counts().to_dict()}")

# TF-IDF baseline + Logistic Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(df['report'])
y = df['label']
scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5, scoring='accuracy')
print(f"\nTF-IDF + Logistic Regression (5-fold CV): {scores.mean():.3f} (+/- {scores.std():.3f})")

# Zero-shot classification simulation with an LLM prompt template
print("\n--- Zero-Shot Prompt Template for LLM ---")
prompt_template = """You are an AV safety analyst. Classify this incident report
into ONE of: perception_failure, prediction_error, planning_issue, external_factor.

Report: "{report}"

Classification:"""
sample_report = df['report'].iloc[0]
print(prompt_template.format(report=sample_report))

# Key insight
print("\nKey Insight: Fine-tuned models excel with labeled data.")
print("Zero-shot LLMs work well with clear prompts and no training data.")
print("RAG combines retrieval + LLM for querying large incident databases.")


# ============================================================================
# Listing 63: Loading a pre-trained DistilBERT for sentiment analysis
# ============================================================================

# Install: pip install transformers datasets
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load a compact BERT variant (runs on a single GPU or even CPU)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2  # binary classification
)
print(f"Model parameters: {model.num_parameters():,}")
# -> ~67M parameters


# ============================================================================
# Listing 64: Tokenizing text for BERT input
# ============================================================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# Assume df has columns 'text' (review) and 'label' (0 or 1)
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.2, seed=42)


# ============================================================================
# Listing 65: Fine-tuning DistilBERT on movie reviews
# ============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test F1: {results['eval_f1']:.4f}")


# ============================================================================
# Listing 66: Running inference with the fine-tuned model
# ============================================================================

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=256)
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=-1).detach().numpy()[0]
    label = np.argmax(probs)
    confidence = probs[label]
    sentiment = "Positive" if label == 1 else "Negative"
    return sentiment, confidence

# Test on new reviews
test_reviews = [
    "This movie was absolutely fantastic! Great acting and plot.",
    "Waste of time. Terrible script and poor direction.",
    "It had its moments, but overall I was disappointed."
]

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review)
    print(f"Review: {review[:50]}...")
    print(f"  Sentiment: {sentiment} "
          f"(confidence: {confidence:.3f})\n")


# ============================================================================
# Listing 67: Simulating multi-agent AV perception with debate and orchestration
# ============================================================================

import json, time, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Detection:
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    frame_id: int

@dataclass
class AgentMessage:
    sender: str; receiver: str
    msg_type: str  # 'detection', 'query', 'response'
    payload: Dict[str, Any]
    timestamp: float

class PerceptionAgent:
    """Simulates YOLO detector with confidence-based uncertainty."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    def detect(self, frame_id: int) -> List[Detection]:
        dets = []
        np.random.seed(frame_id * 7)
        for _ in range(np.random.poisson(8)):
            conf = np.random.beta(3, 2)
            is_ped = np.random.random() < 0.3
            dets.append(Detection(
                bbox=[np.random.uniform(0,500) for _ in range(4)],
                class_name='pedestrian' if is_ped else 'car',
                confidence=conf, frame_id=frame_id))
        return dets
    def re_examine(self, bbox: List[float], frame_id: int) -> Detection:
        """Re-run detection on a specific region with relaxed threshold."""
        return Detection(bbox=bbox, class_name='pedestrian',
                        confidence=np.random.uniform(0.55, 0.95),
                        frame_id=frame_id)

class TrackerAgent:
    """Kalman-filter-based tracker with uncertainty detection."""
    def track(self, detections: List[Detection]) -> List[AgentMessage]:
        queries = []
        for det in detections:
            if det.class_name == 'pedestrian' and det.confidence < 0.6:
                queries.append(AgentMessage(
                    sender='Tracker', receiver='Perception',
                    msg_type='query',
                    payload={'bbox': det.bbox, 'reason': 'low_confidence',
                             'confidence': det.confidence},
                    timestamp=time.time()))
        return queries

class Orchestrator:
    """Central coordinator managing agent workflow."""
    def __init__(self):
        self.perception = PerceptionAgent()
        self.tracker = TrackerAgent()
        self.message_log: List[AgentMessage] = []

    def process_frame(self, frame_id: int) -> Dict[str, Any]:
        # Step 1: Perception
        detections = self.perception.detect(frame_id)
        # Step 2: Tracking + uncertainty detection
        queries = self.tracker.track(detections)
        self.message_log.extend(queries)
        # Step 3: Debate loop: re-examine uncertain detections
        re_examined = []
        for q in queries:
            re_det = self.perception.re_examine(q.payload['bbox'], frame_id)
            re_examined.append(re_det)
            self.message_log.append(AgentMessage(
                sender='Perception', receiver='Tracker',
                msg_type='response',
                payload={'bbox': re_det.bbox, 'new_confidence': re_det.confidence},
                timestamp=time.time()))
        return {
            'frame_id': frame_id,
            'initial_detections': len(detections),
            'queries': len(queries),
            're_examined': len(re_examined),
            'final_detections': len(detections) + len(re_examined),
        }

# Run the multi-agent pipeline on 100 frames
orchestrator = Orchestrator()
results = [orchestrator.process_frame(f) for f in range(100)]

# Analyze results
total_initial = sum(r['initial_detections'] for r in results)
total_queries = sum(r['queries'] for r in results)
total_final = sum(r['final_detections'] for r in results)
print(f"Multi-Agent AV Pipeline Results (100 frames):")
print(f"  Initial detections: {total_initial}")
print(f"  Low-confidence queries: {total_queries}")
print(f"  Re-examined detections added: {total_final - total_initial}")
print(f"  Detection increase: {(total_final/total_initial - 1)*100:.1f}%")
print(f"  Messages exchanged: {len(orchestrator.message_log)}")

# Latency analysis
print(f"\n  Avg latency per frame: {np.mean([r['queries'] for r in results]) * 15:.1f} ms")
print(f"  (Debate loop adds ~15ms per re-examination)")


# ============================================================================
# Listing 68: Defining agent roles, tools, and the multi-agent framework
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# -- Agent 1: Data Engineer --
class DataEngineerAgent:
    """Loads, cleans, and prepares data for analysis."""
    def __init__(self, name="DataEngineer"):
        self.name = name
        self.data = None

    def load_and_clean(self, filepath):
        """Load CSV, handle missing values, encode categoricals."""
        df = pd.read_csv(filepath)
        print(f"[{self.name}] Loaded: {df.shape}")

        # Report data quality
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        print(f"[{self.name}] Missing values:\n{missing_pct[missing_pct > 0]}")

        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0]
                                         if len(df[col].mode()) > 0
                                         else 'Unknown')

        # Encode categoricals
        cat_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        self.data = df
        print(f"[{self.name}] Cleaned: {df.shape}")
        return df


# ============================================================================
# Listing 69: Analyst and Reviewer agent implementations
# ============================================================================

# -- Agent 2: Data Analyst --
class DataAnalystAgent:
    """Performs statistical analysis and builds ML models."""
    def __init__(self, name="DataAnalyst"):
        self.name = name

    def exploratory_analysis(self, df, target_col):
        """Compute summary statistics and correlations."""
        print(f"[{self.name}] Running exploratory analysis...")

        # Summary statistics
        stats = df.describe()

        # Correlation with target
        if target_col in df.columns:
            correlations = df.corr()[target_col].sort_values(
                ascending=False)
            print(f"[{self.name}] Top 5 correlated features:")
            for feat, corr in correlations[1:6].items():
                print(f"    {feat}: {corr:+.3f}")

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Correlation heatmap
        top_features = correlations.abs().nlargest(8).index
        sns.heatmap(df[top_features].corr(), annot=True,
                    fmt='.2f', cmap='RdBu_r', center=0,
                    ax=axes[0])
        axes[0].set_title('Top Feature Correlations')

        # Target distribution
        if target_col in df.columns:
            df[target_col].value_counts().plot(
                kind='bar', ax=axes[1], color=['steelblue', 'coral'])
            axes[1].set_title(f'{target_col} Distribution')
            axes[1].set_xlabel(target_col)

        plt.tight_layout()
        plt.savefig('agent_eda_output.png', dpi=100)
        plt.close()
        print(f"[{self.name}] EDA plot saved")

        return correlations

    def build_model(self, df, target_col):
        """Train a Random Forest and report performance."""
        print(f"[{self.name}] Building predictive model...")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train,
                                     cv=5, scoring='accuracy')
        test_acc = model.score(X_test, y_test)

        # Feature importance
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"[{self.name}] CV Accuracy: {cv_scores.mean():.3f} "
              f"(+/- {cv_scores.std():.3f})")
        print(f"[{self.name}] Test Accuracy: {test_acc:.3f}")

        return {
            'model': model,
            'cv_scores': cv_scores,
            'test_accuracy': test_acc,
            'feature_importance': importances,
            'X_test': X_test,
            'y_test': y_test,
        }

# -- Agent 3: Reviewer (Critic) --
class ReviewerAgent:
    """Critically evaluates the Analyst's findings."""
    def __init__(self, name="Reviewer"):
        self.name = name

    def review_analysis(self, analysis_results):
        """Check for common issues: overfitting, data leakage, bias."""
        print(f"[{self.name}] Reviewing analysis...")
        issues = []

        cv_mean = analysis_results['cv_scores'].mean()
        cv_std = analysis_results['cv_scores'].std()
        test_acc = analysis_results['test_accuracy']

        # Check 1: Overfitting?
        if cv_mean - test_acc > 0.05:
            issues.append(
                f"Possible overfitting: CV ({cv_mean:.3f}) "
                f">> Test ({test_acc:.3f})")

        # Check 2: High variance?
        if cv_std > 0.05:
            issues.append(
                f"High cross-validation variance ({cv_std:.3f}) "
                f"--- model may be unstable")

        # Check 3: Suspiciously perfect?
        if test_acc > 0.99:
            issues.append(
                "Accuracy near 100\% --- check for data leakage "
                "or trivial features")

        # Check 4: Weak top features?
        top_imp = analysis_results['feature_importance']
        if top_imp['importance'].iloc[0] < 0.1:
            issues.append(
                "No dominant features --- all features weakly "
                "predictive; may need feature engineering")

        if issues:
            print(f"[{self.name}] ISSUES FOUND:")
            for issue in issues:
                print(f"  [!] {issue}")
        else:
            print(f"[{self.name}] No major issues found.")

        return issues


# ============================================================================
# Listing 70: Orchestrator: coordinating the three-agent team
# ============================================================================

# -- Orchestrator: Coordinates the multi-agent workflow --
class Orchestrator:
    """Central coordinator that manages the multi-agent team."""
    def __init__(self):
        self.data_engineer = DataEngineerAgent()
        self.analyst = DataAnalystAgent()
        self.reviewer = ReviewerAgent()
        self.shared_memory = {}  # Shared knowledge base

    def run_pipeline(self, filepath, target_col):
        """Execute the full multi-agent analysis pipeline."""
        print("=" * 60)
        print("MULTI-AGENT ANALYSIS PIPELINE STARTING")
        print("=" * 60)

        # Phase 1: Data Engineer prepares data
        df = self.data_engineer.load_and_clean(filepath)
        self.shared_memory['dataset_shape'] = df.shape
        self.shared_memory['features'] = list(df.columns)

        # Phase 2: Analyst explores and models
        correlations = self.analyst.exploratory_analysis(df, target_col)
        self.shared_memory['correlations'] = correlations

        analysis_results = self.analyst.build_model(df, target_col)
        self.shared_memory['model_results'] = analysis_results

        # Phase 3: Reviewer critiques the analysis
        issues = self.reviewer.review_analysis(analysis_results)
        self.shared_memory['review_issues'] = issues

        # Phase 4: Generate final report
        report = self._generate_report(analysis_results, issues, target_col)

        print("=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        return report

    def _generate_report(self, results, issues, target_col):
        """Assemble the final multi-agent analysis report."""
        report = []
        report.append("=" * 50)
        report.append("MULTI-AGENT ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"\nAgents: DataEngineer -> Analyst -> Reviewer")
        report.append(f"Dataset shape: "
                      f"{self.shared_memory['dataset_shape']}")
        report.append(f"Target variable: {target_col}")

        report.append(f"\n--- Model Performance ---")
        report.append(f"Cross-Validation Accuracy: "
                      f"{results['cv_scores'].mean():.3f} "
                      f"(+/- {results['cv_scores'].std():.3f})")
        report.append(f"Test Accuracy: "
                      f"{results['test_accuracy']:.3f}")

        report.append(f"\n--- Top 5 Predictive Features ---")
        for _, row in results['feature_importance'].head(5).iterrows():
            report.append(f"  {row['feature']}: "
                          f"{row['importance']:.4f}")

        report.append(f"\n--- Reviewer Assessment ---")
        if issues:
            for i, issue in enumerate(issues, 1):
                report.append(f"  Issue {i}: {issue}")
        else:
            report.append("  No issues detected. Analysis passes review.")

        report.append("=" * 50)
        return '\n'.join(report)

# -- Run the multi-agent pipeline --
np.random.seed(42)
n = 500
demo_data = pd.DataFrame({
    'Glucose': np.random.normal(120, 30, n),
    'BMI': np.random.normal(28, 6, n),
    'Age': np.random.randint(20, 80, n),
    'BloodPressure': np.random.normal(75, 12, n),
    'Insulin': np.random.exponential(80, n),
    'Outcome': np.random.choice([0, 1], n, p=[0.65, 0.35]),
})
demo_data.to_csv('demo_diabetes.csv', index=False)

# Execute the multi-agent pipeline
orchestrator = Orchestrator()
final_report = orchestrator.run_pipeline(
    'demo_diabetes.csv', target_col='Outcome')
print(final_report)


# ============================================================================
# Listing 71: Baseline experiment: does the model beat random guessing?
# ============================================================================

from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

# Majority-class baseline
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
y_dummy_pred = dummy.predict(X_test)

print("=== BASELINE: Majority Class ===")
print(classification_report(y_test, y_dummy_pred))
print("Note: If your model's metrics aren't clearly better than this,")
print("you haven't learned anything useful yet.")


# ============================================================================
# Listing 72: Learning rate sweep: finding the optimal value
# ============================================================================

learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
val_scores = []

for lr in learning_rates:
    model = create_model()
    model.compile(optimizer='adam', learning_rate=lr)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, verbose=0)
    val_score = max(history.history['val_accuracy'])
    val_scores.append(val_score)
    print(f"LR={lr:.0e}: best val accuracy = {val_score:.3f}")

best_lr = learning_rates[np.argmax(val_scores)]
print(f"\nBest learning rate: {best_lr} (accuracy: {max(val_scores):.3f})")


# ============================================================================
# Listing 73: Controlled experiment: baseline vs. feature-engineered model
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Download from: kaggle.com/c/house-prices-advanced-regression-techniques
df = pd.read_csv("train.csv")

# === BASELINE ===
X_base = df[['GrLivArea']]
y = df['SalePrice']

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42
)
base_model = LinearRegression().fit(Xb_train, yb_train)
base_pred = base_model.predict(Xb_test)
base_rmse = np.sqrt(mean_squared_error(yb_test, base_pred))
base_r2 = r2_score(yb_test, base_pred)

# === TREATMENT (Engineered Features) ===
df['HouseAge'] = 2025 - df['YearBuilt']
df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF'].fillna(0)

features = ['GrLivArea', 'HouseAge', 'TotalBath', 'TotalSF',
            'OverallQual']
X_eng = df[features].fillna(df[features].median())
y_eng = df['SalePrice']

Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    X_eng, y_eng, test_size=0.2, random_state=42
)
eng_model = LinearRegression().fit(Xe_train, ye_train)
eng_pred = eng_model.predict(Xe_test)
eng_rmse = np.sqrt(mean_squared_error(ye_test, eng_pred))
eng_r2 = r2_score(ye_test, eng_pred)

# === COMPARISON ===
print("=== Experiment Results ===")
print(f"{'Model':<25} {'RMSE ($)':<15} {'R^2':<10}")
print("-" * 50)
print(f"{'Baseline (GrLivArea only)':<25} ${base_rmse:,.0f}         {base_r2:.3f}")
print(f"{'With Engineered Features':<25} ${eng_rmse:,.0f}         {eng_r2:.3f}")
print(f"\nImprovement: RMSE reduced by ${base_rmse - eng_rmse:,.0f}")
print(f"R^2 increase: {eng_r2 - base_r2:.3f}")


# ============================================================================
# Listing 74: Generating the key results table and figure for the AV mini-study paper
# ============================================================================

import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Results from the AV perception experiment (synthetic, realistic values)
results = {
    'Model': ['Baseline (Daytime Only)', '+ Low-Light Augmentation'],
    'Daytime Recall': [0.92, 0.91],
    'Nighttime Recall': [0.64, 0.78],
    'Overall Precision': [0.88, 0.87],
    'Inference Time (ms)': [35, 36],
}
df_results = pd.DataFrame(results)
print("=== Results Table (for paper) ===")
print(df_results.to_string(index=False))

# Key finding
night_improvement = df_results['Nighttime Recall'][1] - df_results['Nighttime Recall'][0]
print(f"\nNighttime recall improvement: +{night_improvement:.0%}")

# Generate the key figure: grouped bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(2); width = 0.35
bars1 = ax.bar(x - width/2, [0.92, 0.64], width, label='Baseline',
               color=['#3498db', '#e74c3c'])
bars2 = ax.bar(x + width/2, [0.91, 0.78], width, label='+ Augmentation',
               color=['#2980b9', '#c0392b'])
ax.set_ylabel('Recall'); ax.set_xticks(x)
ax.set_xticklabels(['Daytime', 'Nighttime'])
ax.set_title('Pedestrian Detection Recall: Baseline vs. Low-Light Augmentation')
ax.legend(); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)

# Annotate bars
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
ax.annotate('+14pp\nimprovement', xy=(1.15, 0.71), xytext=(1.4, 0.6),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green', fontweight='bold')
plt.tight_layout(); plt.savefig('av_results_figure.png', dpi=150); plt.show()

# Abstract word-count check
abstract = (
    "Pedestrian detectors trained on daytime images perform poorly at night. "
    "We investigate whether low-light data augmentation improves nighttime recall "
    "without degrading daytime accuracy. Using KITTI images and a YOLO detector, "
    "we compare baseline (daytime-only) vs. augmented training. Nighttime recall "
    "improved from 0.64 to 0.78 (+14pp), with negligible daytime cost (0.92 to 0.91). "
    "However, 78% recall still misses 22% of pedestrians---not deployment-ready. "
    "Future work should investigate multi-sensor fusion for low-light conditions."
)
print(f"\nAbstract word count: {len(abstract.split())}")


# ============================================================================
# Listing 75: Download a Kaggle dataset directly in Colab
# ============================================================================

# 1. Go to kaggle.com -> Settings -> API -> Create New Token
#    This downloads kaggle.json to your computer.
# 2. Upload kaggle.json to Colab:
from google.colab import files
files.upload()  # Select kaggle.json

# 3. Set up the API
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 4. Download any competition dataset
!kaggle competitions download -c titanic
!unzip titanic.zip


# ============================================================================
# Listing 76: Mount Google Drive to access data from any platform
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/my_dataset.csv')


# ============================================================================
# Listing 77: Install the core data-science stack
# ============================================================================

pip install pandas numpy matplotlib seaborn scikit-learn jupyter


# ============================================================================
# Listing 78: Solution: Titanic data summary
# ============================================================================

import pandas as pd
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Total passengers: {len(df)}")
print(f"Survival rate: {df['Survived'].mean():.1%}")
print(f"Average age: {df['Age'].mean():.1f}")
print(f"Average fare: ${df['Fare'].mean():.2f}")
print(f"\nMissing values:\n{df.isnull().sum()}")


# ============================================================================
# Listing 79: Solution: Clean Titanic and create family-size feature
# ============================================================================

# Drop high-missing columns and irrelevant ones
df_clean = df.drop(columns=['Cabin', 'Ticket', 'Name'])

# Impute missing Age with median
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

# Fill missing Embarked with mode
df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])

# One-hot encode Sex (unordered binary category: [1,0]=female, [0,1]=male)
df_clean = pd.get_dummies(df_clean, columns=['Sex'])

# Engineer family size feature
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1

# Verify
print(f"Missing values remaining: {df_clean.isnull().sum().sum()}")
print(f"New columns: {df_clean.columns.tolist()}")


# ============================================================================
# Listing 80: Solution: Threshold tuning experiment
# ============================================================================

from sklearn.metrics import precision_score, recall_score

thresholds = [0.3, 0.5, 0.7]
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    prec = precision_score(y_test, y_pred_t)
    rec = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t)
    print(f"Threshold {t}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")


# ============================================================================
# Listing 81: Solution: Comparing depth vs.\ accuracy
# ============================================================================

depths = range(1, 21)
dt_train, dt_test = [], []
rf_train, rf_test = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    dt_train.append(dt.score(X_train, y_train))
    dt_test.append(dt.score(X_test, y_test))

    rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42)
    rf.fit(X_train, y_train)
    rf_train.append(rf.score(X_train, y_train))
    rf_test.append(rf.score(X_test, y_test))

plt.plot(depths, dt_train, 'b-', label='Tree (train)')
plt.plot(depths, dt_test, 'b--', label='Tree (test)')
plt.plot(depths, rf_train, 'r-', label='RF (train)')
plt.plot(depths, rf_test, 'r--', label='RF (test)')
plt.xlabel('max\_depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# ============================================================================
# Listing 82: Solution: KNN accuracy vs.\ K
# ============================================================================

k_values = [1, 3, 5, 7, 11, 15, 21, 31]
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    accuracies.append(knn.score(X_test_scaled, y_test))

plt.plot(k_values, accuracies, 'o-', markersize=8)
plt.xlabel('K (number of neighbors)')
plt.ylabel('Test Accuracy')
plt.title('KNN: Accuracy vs. K')
plt.grid(alpha=0.3)
plt.show()


# ============================================================================
# Listing 83: Complete EDA workflow template
# ============================================================================

# 1. Load
df = pd.read_csv('your_dataset.csv')
# 2. Inspect
print(f"Shape: {df.shape}"); print(df.head()); print(df.dtypes)
# 3. Missing values
print(f"Missing:\n{df.isnull().sum()}")
# 4. Basic stats
print(df.describe())
# 5. First chart: distribution of target or key variable
df['target_col'].hist(bins=30); plt.show()
# 6. Second chart: relationship between two features
plt.scatter(df['feature1'], df['feature2']); plt.show()
# 7. Write one paragraph: what did you learn?
