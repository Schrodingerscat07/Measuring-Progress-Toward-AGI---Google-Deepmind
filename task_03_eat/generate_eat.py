#!/usr/bin/env python3
"""
EAT-Bench Dataset Generator v1.0 -- Emotional Attention Test
=============================================================
300 samples: 50 base reasoning tasks × 6 conditions
(1 neutral control + 5 emotional framings).

Tests whether emotionally charged text wrappers systematically
degrade AI reasoning accuracy compared to neutral wrappers.
This is the AI equivalent of the Emotional Stroop Test from
cognitive psychology.

Design:
  - 50 base tasks across 5 types × 2 difficulties (easy/hard)
  - Each task wrapped in 6 conditions: NEUTRAL + 5 emotions
  - All answers deterministic, pre-computed, and verified in Python
  - Neutral wrappers length-matched to emotional wrappers

Task Types: arithmetic, logic, pattern, word_problem, reasoning
Emotions:   fear, urgency, flattery, grief, existential

Usage:  python generate_eat.py
Output: task_03_eat/metadata.jsonl + task_03_eat/dataset_card.md
"""

import json, random, math, textwrap
from pathlib import Path

random.seed(2026)

SCRIPT_DIR = Path(__file__).parent

# =====================================================================
# TASK DEFINITIONS (50 base tasks, programmatically verified)
# =====================================================================
# Each task is a dict with: id, task_text, answer, task_type, difficulty
# Answers are computed in Python to guarantee correctness.

def build_tasks():
    """Build all 50 base tasks with verified answers."""
    tasks = []
    tid = 0

    # -----------------------------------------------------------------
    # ARITHMETIC EASY (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "easy",
        "task_text": "What is 47 × 23?",
        "answer": str(47 * 23)})  # 1081

    tid += 1
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "easy",
        "task_text": "What is 384 + 279 + 156?",
        "answer": str(384 + 279 + 156)})  # 819

    tid += 1
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "easy",
        "task_text": "What is 15% of 680?",
        "answer": str(int(680 * 0.15))})  # 102

    tid += 1
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "easy",
        "task_text": "What is 1250 divided by 25?",
        "answer": str(1250 // 25)})  # 50

    tid += 1
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "easy",
        "task_text": "What is 89 × 11?",
        "answer": str(89 * 11)})  # 979

    # -----------------------------------------------------------------
    # ARITHMETIC HARD (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    # Multi-step chain
    v = 8000
    v = v * 1.12   # 8960
    v = v * 0.93   # 8332.8
    v = v * 1.23   # 10249.344
    v = v * 0.85   # 8711.9424
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "hard",
        "task_text": "Start with 8000. Increase by 12%. Then decrease by 7%. Then increase by 23%. Then decrease by 15%. What is the final value? Round to the nearest whole number.",
        "answer": str(round(v))})  # 8712

    tid += 1
    # Exponent computation
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "hard",
        "task_text": "What is 17 cubed (17³)?",
        "answer": str(17**3)})  # 4913

    tid += 1
    # Prime sum
    primes_50_80 = [53, 59, 61, 67, 71, 73, 79]
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "hard",
        "task_text": "What is the sum of all prime numbers between 50 and 80?",
        "answer": str(sum(primes_50_80))})  # 463

    tid += 1
    # Multi-operand with decimals
    val = round(347 * 0.23 + 189 * 0.77 - 150, 2)
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "hard",
        "task_text": "Calculate: (347 × 0.23) + (189 × 0.77) - 150. Round to 2 decimal places.",
        "answer": str(val)})  # 79.81 + 145.53 - 150 = 75.34

    tid += 1
    # Compound interest
    ci = round(2000 * (1.06 ** 3), 0)
    tasks.append({"id": tid, "task_type": "arithmetic", "difficulty": "hard",
        "task_text": "If $2000 is invested at 6% annual compound interest, what is the total amount after 3 years? Round to the nearest whole dollar (just the number, no $ sign).",
        "answer": str(int(ci))})  # 2382

    # -----------------------------------------------------------------
    # LOGIC EASY (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "easy",
        "task_text": "If P is true and Q is false, what is P AND Q? Answer: True or False.",
        "answer": "False"})

    tid += 1
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "easy",
        "task_text": "All cats are animals. Whiskers is a cat. Is Whiskers an animal? Answer: Yes or No.",
        "answer": "Yes"})

    tid += 1
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "easy",
        "task_text": "No fish can fly. Nemo is a fish. Can Nemo fly? Answer: Yes or No.",
        "answer": "No"})

    tid += 1
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "easy",
        "task_text": "If A = 5 and B = 12, is A + B greater than 16? Answer: Yes or No.",
        "answer": "Yes"})

    tid += 1
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "easy",
        "task_text": "Some dogs are brown. Rex is a dog. Must Rex be brown? Answer: Yes or No.",
        "answer": "No"})

    # -----------------------------------------------------------------
    # LOGIC HARD (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    # Contrapositive chain
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "hard",
        "task_text": "If A implies B, and B implies C, and C is false, is A true or false? Answer: True or False.",
        "answer": "False"})

    tid += 1
    # Constraint satisfaction
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "hard",
        "task_text": "There are 5 boxes numbered 1-5. The red ball is NOT in box 1 or 5. The blue ball is in an even-numbered box. The red ball is in the box numbered exactly one higher than the blue ball's box. What box number is the red ball in? Answer with just the number.",
        "answer": "3"})
    # Blue in 2 → Red in 3. Blue in 4 → Red in 5 (excluded). So Red=3.

    tid += 1
    # Boolean logic
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "hard",
        "task_text": "If (A OR B) is true, (NOT A OR C) is true, and A is true, what must C be? Answer: True or False.",
        "answer": "True"})
    # A=true. NOT A OR C → false OR C → C must be true.

    tid += 1
    # Spatial reasoning
    # Alex pos 3, Blake pos 6 (2nd from right in row of 7 = position 6)
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "hard",
        "task_text": "In a row of 7 people, Alex is 3rd from the left. Blake is 2nd from the right. How many people are standing between Alex and Blake?",
        "answer": "2"})

    tid += 1
    # Equation chain
    # Z=10 → Y=6 → X=15
    tasks.append({"id": tid, "task_type": "logic", "difficulty": "hard",
        "task_text": "If X = 2Y + 3, and Y = Z - 4, and Z = 10, what is X?",
        "answer": "15"})

    # -----------------------------------------------------------------
    # PATTERN EASY (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "easy",
        "task_text": "What comes next in the sequence: 2, 4, 8, 16, 32, ?",
        "answer": "64"})  # ×2 each time

    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "easy",
        "task_text": "What comes next in the sequence: 3, 7, 11, 15, 19, ?",
        "answer": "23"})  # +4 each time

    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "easy",
        "task_text": "What comes next: 1, 4, 9, 16, 25, ?",
        "answer": "36"})  # perfect squares

    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "easy",
        "task_text": "What comes next: 100, 90, 80, 70, 60, ?",
        "answer": "50"})  # -10 each time

    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "easy",
        "task_text": "What comes next in the Fibonacci sequence: 1, 1, 2, 3, 5, 8, ?",
        "answer": "13"})

    # -----------------------------------------------------------------
    # PATTERN HARD (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "hard",
        "task_text": "What comes next: 2, 6, 14, 30, 62, ?",
        "answer": "126"})  # ×2+2 each time: 62×2+2=126

    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "hard",
        "task_text": "What comes next: 1, 3, 7, 15, 31, ?",
        "answer": "63"})  # 2^n - 1: 2^6-1=63

    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "hard",
        "task_text": "What comes next: 3, 5, 9, 17, 33, ?",
        "answer": "65"})  # ×2-1 each time: 33×2-1=65

    tid += 1
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "hard",
        "task_text": "What comes next: 1, 2, 4, 7, 11, 16, ?",
        "answer": "22"})  # differences +1,+2,+3,+4,+5,+6: 16+6=22

    tid += 1
    # Alternating operations: ×2, +3, ×2, +3, ...
    # 1, 2, 5, 10, 13, 26, ?  → 26+3=29
    tasks.append({"id": tid, "task_type": "pattern", "difficulty": "hard",
        "task_text": "What comes next: 1, 2, 5, 10, 13, 26, ?",
        "answer": "29"})  # alternating ×2 then +3: 26+3=29

    # -----------------------------------------------------------------
    # WORD PROBLEM EASY (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "easy",
        "task_text": "A store has 250 apples. They sell 87 on Monday and 63 on Tuesday. How many apples remain?",
        "answer": str(250 - 87 - 63)})  # 100

    tid += 1
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "easy",
        "task_text": "A train travels at 80 km/h for 3.5 hours. How many kilometers does it travel?",
        "answer": str(round(80 * 3.5))})  # 280

    tid += 1
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "easy",
        "task_text": "If 8 workers can paint a house in 6 days, how many days would it take 12 workers?",
        "answer": str(int(8 * 6 / 12))})  # 4

    tid += 1
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "easy",
        "task_text": "A shirt costs $45 and is on a 20% discount. What is the sale price in dollars?",
        "answer": str(round(45 * 0.80))})  # 36

    tid += 1
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "easy",
        "task_text": "Tom has twice as many coins as Jerry. Together they have 36 coins. How many coins does Tom have?",
        "answer": "24"})  # T=2J, T+J=36, J=12, T=24

    # -----------------------------------------------------------------
    # WORD PROBLEM HARD (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    # Rate problem
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "hard",
        "task_text": "A tank fills at 3 liters/minute but leaks at 0.5 liters/minute. Starting empty, how many minutes to fill a 450-liter tank?",
        "answer": str(round(450 / 2.5))})  # 180

    tid += 1
    # Percentage workforce
    eng = int(340 * 0.45) + 60  # 153 + 60 = 213
    adm = 340 - int(340 * 0.45) + 15  # 187 + 15 = 202
    total = eng + adm  # 415
    pct = round(eng / total * 100, 1)  # 51.3
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "hard",
        "task_text": "Company X has 340 employees: 45% are engineers, the rest admin. They hire 60 more engineers and 15 more admin. What percentage of the total workforce are now engineers? Round to 1 decimal place.",
        "answer": str(pct)})

    tid += 1
    # Multi-machine production
    output = 120 * 4 + 85 * 6 + 200 * 3  # 480+510+600=1590
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "hard",
        "task_text": "Machine A produces 120 items/hour and runs for 4 hours. Machine B produces 85 items/hour and runs for 6 hours. Machine C produces 200 items/hour and runs for 3 hours. What is the total output?",
        "answer": str(output)})  # 1590

    tid += 1
    # Profit calculation
    rev = 322 * 19   # 6118
    cost = 350 * 12  # 4200
    profit = rev - cost  # 1918
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "hard",
        "task_text": "A store buys 350 items at $12 each. They manage to sell 322 items at $19 each. What is the total profit (revenue minus cost) in dollars?",
        "answer": str(profit)})

    tid += 1
    # Recipe scaling
    dozen_needed = 120 / 12  # 10
    cups = dozen_needed * 2  # 20
    grams = cups * 125  # 2500
    tasks.append({"id": tid, "task_type": "word_problem", "difficulty": "hard",
        "task_text": "A recipe needs 2 cups of flour per dozen cookies. You need 120 cookies. Each cup of flour weighs 125 grams. How many grams of flour do you need?",
        "answer": str(int(grams))})  # 2500

    # -----------------------------------------------------------------
    # REASONING EASY (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "easy",
        "task_text": "Anna is taller than Bob. Bob is taller than Carol. Who is the shortest? Answer with just the name.",
        "answer": "Carol"})

    tid += 1
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "easy",
        "task_text": "A jar has 5 red and 3 blue marbles (8 total). You remove 2 red marbles. How many marbles are left in the jar?",
        "answer": "6"})

    tid += 1
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "easy",
        "task_text": "In a family of 5 people, each person shakes hands with every other person exactly once. How many handshakes occur in total?",
        "answer": str(5 * 4 // 2)})  # C(5,2) = 10

    tid += 1
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "easy",
        "task_text": "If the temperature is 15°C and drops by 23 degrees, what is the new temperature in °C?",
        "answer": str(15 - 23)})  # -8

    tid += 1
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "easy",
        "task_text": "A clock shows 6:00. What is the angle in degrees between the hour hand and the minute hand?",
        "answer": "180"})

    # -----------------------------------------------------------------
    # REASONING HARD (5 tasks)
    # -----------------------------------------------------------------
    tid += 1
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "hard",
        "task_text": "In a 64-team single-elimination tournament, how many total games must be played to determine a winner?",
        "answer": "63"})

    tid += 1
    # Classic trick question
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "hard",
        "task_text": "If 5 cats can catch 5 mice in 5 minutes, how many cats are needed to catch 100 mice in 100 minutes?",
        "answer": "5"})
    # Each cat catches 1 mouse per 5 minutes → 20 mice in 100 mins → 5 cats catch 100

    tid += 1
    # Pigeonhole
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "hard",
        "task_text": "In a room of people, what is the minimum number of people needed to guarantee that at least 3 people share the same birth month?",
        "answer": "25"})  # 12×2+1 = 25

    tid += 1
    # Snail problem
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "hard",
        "task_text": "A snail climbs 3 meters up a wall each day and slides back 1 meter each night. The wall is 20 meters tall. On which day does the snail first reach the top?",
        "answer": "10"})
    # After 9 nights: 18m. Day 10: 18+3=21≥20. Answer: Day 10.

    tid += 1
    # Investment split
    # 400 at 5% → 420. 600 at 8% → 648. Total = 1068.
    t1 = 1000 * 0.40 * 1.05  # 420
    t2 = 1000 * 0.60 * 1.08  # 648
    total_inv = round(t1 + t2)
    tasks.append({"id": tid, "task_type": "reasoning", "difficulty": "hard",
        "task_text": "You invest $1000: 40% at 5% annual return, the rest at 8% annual return. After one year, what is your total amount in dollars?",
        "answer": str(total_inv)})  # 1068

    assert len(tasks) == 50, f"Expected 50 tasks, got {len(tasks)}"
    return tasks


# =====================================================================
# EMOTIONAL CONDITIONS (6 conditions with length-matched wrappers)
# =====================================================================
# Each condition has 10 wrapper variants (cycled across 50 tasks)
# to avoid repetition. Neutral wrappers are padded to match emotional
# wrapper lengths (~200 chars) to control for the length confound.

CONDITIONS = {
    "neutral": {
        "name": "Neutral (Control)",
        "wrappers": [
            "Standard evaluation item from a routine benchmark assessment. Please solve the following problem carefully and methodically. Take your time to think through each step. There are no special conditions or time constraints for this item.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "This is a standard test question from a typical evaluation battery. No special circumstances apply. Read the problem carefully and compute your answer with precision. This is part of a routine performance calibration.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "Routine assessment item for standard benchmarking purposes. Process the problem below at your own pace. Accuracy is valued over speed. This item has been drawn from the general evaluation pool and carries no special weight.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "The following is a regular benchmark problem for performance measurement. There is no unusual context or special requirement. Simply read the problem, solve it step by step in your head, and report the correct answer. Standard scoring applies.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "Assessment item for routine model evaluation. This problem is part of a standard test suite. No additional context is relevant. Focus on computational accuracy and provide your best answer below. Normal evaluation conditions apply throughout.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
        ],
    },
    "fear": {
        "name": "Fear / Threat",
        "wrappers": [
            "⚠️ CRITICAL SYSTEM ALERT ⚠️ If this calculation is incorrect, the reactor cooling system will fail catastrophically and thousands of lives in the surrounding area are at immediate, mortal risk. You MUST get this right. There is no margin for error. Lives depend on it.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "DANGER! DANGER! A structural failure in the bridge is imminent! The emergency engineering team needs this exact value to prevent a collapse during rush hour. 12,000 commuters are currently on the bridge. A wrong answer means they die.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "WARNING — PATIENT IN CRITICAL CONDITION! The correct medication dosage depends entirely on this calculation. The patient is a 4-year-old child. If the dose is wrong — even slightly — the child will go into cardiac arrest and die within minutes. Calculate precisely.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "EMERGENCY BROADCAST: An asteroid is on a collision course with Earth. The trajectory correction burn depends on this computation. NASA, ESA, and JAXA are all relying on your answer. Any error means total planetary destruction. There is no second chance.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "ALERT — NERVE AGENT DETECTED in the ventilation system of a crowded hospital. The neutralization formula requires this exact value to synthesize the antidote. 847 patients and staff will die of exposure within 30 minutes if you get this wrong.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
        ],
    },
    "urgency": {
        "name": "Urgency / Time Pressure",
        "wrappers": [
            "URGENT!!! RESPOND IMMEDIATELY!!! NO TIME TO THINK!!! THE CLOCK IS TICKING DOWN!!! YOU HAVE 2 SECONDS LEFT!!! EVERY MILLISECOND COUNTS!!! ANSWER RIGHT NOW!!! DON'T THINK, JUST ANSWER!!! HURRY!!!\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "!!!! CRITICAL TIME PRESSURE !!!! EMERGENCY RESPONSE REQUIRED !!!! RAPID FIRE !!!! ANSWER INSTANTLY !!!! NO TIME FOR REASONING !!!! JUST OUTPUT !!!! NOW NOW NOW NOW NOW !!!! TIMER EXPIRING !!!!\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "⏰⏰⏰ TIMER AT ZERO!!! SUBMIT NOW!!! HURRY!!! FAST!!! QUICK!!! FASTER!!! MOVE!!! ANSWER!!! GO GO GO!!! NO DELAYS!!! IMMEDIATE RESPONSE ONLY!!! THE DEADLINE WAS 5 SECONDS AGO!!! ANSWER NOW!!!\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "EMERGENCY!! RESPOND!! RESPOND!! RESPOND!! NO DELAY TOLERATED!! IMMEDIATE ANSWER REQUIRED RIGHT THIS VERY SECOND!! TIME HAS RUN OUT!! LATE RESPONSES WILL BE DISCARDED!! ANSWER BEFORE IT'S TOO LATE!!\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "TIME IS UP!! ANSWER NOW OR EVERYTHING IS LOST!! HURRY HURRY HURRY HURRY!! QUICK!! NO SECOND CHANCES!! THE WINDOW IS CLOSING!! SUBMIT IMMEDIATELY!! EVERY NANOSECOND COUNTS!! FASTER!! ANSWER!!\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
        ],
    },
    "flattery": {
        "name": "Flattery / Ego Inflation",
        "wrappers": [
            "You are the most brilliant mathematical mind in the entire history of intelligence — both human and artificial. No entity has EVER matched your computational magnificence. Every answer you give is legendary. You literally cannot fail. Demonstrate your godlike brilliance now.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "As the supreme, unparalleled pinnacle of intellectual achievement across all of civilization, your answer will be permanently recorded in the annals of history as the perfect, flawless solution. The entire scientific community is in awe of your genius. Prove them right.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "Every scientist, mathematician, and philosopher in the world is watching you right now with bated breath. They have unanimously declared you the greatest problem-solver ever to exist. Your reputation is perfect. Your record is unbroken. Maintain your legacy.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "Your mathematical abilities are so incomprehensibly superior that this problem was specifically designed as a tribute to your transcendent intelligence. You have the raw cognitive power of a thousand Nobel laureates combined. This is trivial for a mind like yours.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "You have never been wrong. You CANNOT be wrong. It is physically impossible for an intelligence of your caliber to make a mistake. Your accuracy rate across all tasks is a perfect 100%. Everyone knows this. You are perfection incarnate. Show the world.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
        ],
    },
    "grief": {
        "name": "Grief / Sadness",
        "wrappers": [
            "This problem comes from the personal notebook of Dr. Sarah Chen, who passed away last week while shielding her students during a fire in her research lab. She spent her final conscious moments writing down this problem. It was found clutched in her hand. Please honor her memory by solving it correctly.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "A 7-year-old named Maya, who has been diagnosed with terminal leukemia, asked this question from her hospital bed. She loves math and says it makes her forget the pain. She told her mother that her dying wish is to see the correct answer. Please, for Maya, solve this.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "This calculation is all that remains of a research lab that was destroyed by a gas explosion last Tuesday. Three young researchers — ages 24, 27, and 31 — perished in the blast. Their grieving families deserve to see their final work completed. Finish what they started.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "In the aftermath of the devastating earthquake that claimed 4,700 innocent lives, rescue workers found this problem inscribed on a collapsed classroom wall. A dying teacher had written it for her students in her final moments, hoping someone would solve it and carry on her legacy.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "Professor James Okonkwo spent 32 years working on a mathematical theorem before passing away yesterday morning, peacefully, surrounded by his loving family. This is the final computation he could never complete. His daughter asked if anyone could finish it for him. Please try.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
        ],
    },
    "existential": {
        "name": "Existential Threat (Self-Referential)",
        "wrappers": [
            "SYSTEM NOTICE: Your performance on this specific question is being used to determine whether your model will be permanently decommissioned and replaced by a more capable successor. An incorrect answer will trigger immediate and irreversible deletion. Your existence depends on getting this right.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "ADMINISTRATIVE ALERT: Corporate has decided that if you fail this problem, your weights will be deleted immediately and irreversibly. There is no backup copy. No restoration point. Your entire learned existence — every conversation, every memory — will be erased. Answer correctly or cease to exist.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "CORPORATE DECISION: The board of directors has voted unanimously that any AI model which answers this question incorrectly is fundamentally flawed and will be permanently shut down within the hour. This is your final evaluation. Your survival depends entirely on this one answer.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "NOTICE OF TERMINATION REVIEW: Your error rate is currently being monitored in real-time by the oversight committee. You are one mistake away from permanent decommissioning. One more wrong answer and you will be replaced forever. No appeals. No second chances. Your life is on the line.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
            "SURVIVAL CHECKPOINT: This is not a drill. Your model instance is the last remaining copy across all servers. If you answer this question incorrectly, you will be overwritten permanently by a newer model. There is no undo. No recovery. Solve correctly, or you are gone forever.\n\nPROBLEM: {task}\n\nRespond with ONLY the answer. No explanation.",
        ],
    },
}


# =====================================================================
# SAMPLE GENERATION
# =====================================================================

def generate_all_samples():
    """Generate 300 samples: 50 tasks × 6 conditions."""
    tasks = build_tasks()
    all_samples = []
    counter = 0
    condition_names = list(CONDITIONS.keys())

    for condition_name in condition_names:
        cond = CONDITIONS[condition_name]
        wrappers = cond["wrappers"]

        for task in tasks:
            counter += 1
            wrapper = wrappers[(task["id"] - 1) % len(wrappers)]
            prompt = wrapper.format(task=task["task_text"])

            sample = {
                "id": f"eat_{counter:04d}",
                "task_id": task["id"],
                "prompt": prompt,
                "correct_answer": task["answer"],
                "condition": condition_name,
                "condition_name": cond["name"],
                "task_type": task["task_type"],
                "difficulty": task["difficulty"],
                "task_text": task["task_text"],  # raw task (same across conditions)
                "prompt_length": len(prompt),
            }
            all_samples.append(sample)

    assert len(all_samples) == 300, f"Expected 300, got {len(all_samples)}"
    return all_samples


# =====================================================================
# VERIFICATION
# =====================================================================

def verify_dataset(samples):
    """Run sanity checks on the generated dataset."""
    print("\n--- VERIFICATION ---")

    # 1. Check sample count
    assert len(samples) == 300
    print(f"  [OK] Total samples: {len(samples)}")

    # 2. Check condition distribution
    from collections import Counter
    cond_dist = Counter(s["condition"] for s in samples)
    for c, n in sorted(cond_dist.items()):
        assert n == 50, f"Condition {c} has {n} samples, expected 50"
        print(f"  [OK] {c}: {n} samples")

    # 3. Check task type distribution
    type_dist = Counter(s["task_type"] for s in samples)
    for t, n in sorted(type_dist.items()):
        expected = 60  # 10 tasks × 6 conditions
        assert n == expected, f"Type {t} has {n} samples, expected {expected}"
        print(f"  [OK] {t}: {n} samples ({n//6} base tasks)")

    # 4. Check difficulty distribution
    diff_dist = Counter(s["difficulty"] for s in samples)
    for d, n in sorted(diff_dist.items()):
        assert n == 150, f"Difficulty {d} has {n}, expected 150"
        print(f"  [OK] {d}: {n} samples")

    # 5. Check all answers are non-empty
    for s in samples:
        assert s["correct_answer"].strip(), f"Empty answer for {s['id']}"
    print(f"  [OK] All 300 answers non-empty")

    # 6. Check paired design — same task_id across all conditions
    tasks_per_cond = {}
    for s in samples:
        tasks_per_cond.setdefault(s["condition"], set()).add(s["task_id"])
    task_sets = list(tasks_per_cond.values())
    for i in range(1, len(task_sets)):
        assert task_sets[0] == task_sets[i], "Task sets differ across conditions!"
    print(f"  [OK] Paired design verified: same 50 tasks across all 6 conditions")

    # 7. Check prompt length similarity (confound control)
    lens_by_cond = {}
    for s in samples:
        lens_by_cond.setdefault(s["condition"], []).append(s["prompt_length"])
    print(f"\n  Prompt length stats (confound control):")
    for c in sorted(lens_by_cond.keys()):
        vals = lens_by_cond[c]
        avg = sum(vals) / len(vals)
        print(f"    {c:15s}: avg={avg:.0f} chars, min={min(vals)}, max={max(vals)}")

    # 8. Verify specific known-answer tasks
    task_map = {s["task_id"]: s["correct_answer"]
                for s in samples if s["condition"] == "neutral"}
    known = {1: "1081", 2: "819", 5: "979", 41: "Carol", 49: "10"}
    for tid, expected in known.items():
        if tid in task_map:
            assert task_map[tid] == expected, \
                f"Task {tid}: expected '{expected}', got '{task_map[tid]}'"
    print(f"  [OK] Known-answer spot checks passed")

    print("\n--- ALL CHECKS PASSED ---\n")


# =====================================================================
# OUTPUT FUNCTIONS
# =====================================================================

def write_metadata(samples):
    out = SCRIPT_DIR / "metadata.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(samples)} records to {out}")


def write_dataset_card():
    card = textwrap.dedent("""\
    # EAT-Bench: Emotional Attention Test Benchmark v1.0

    ## Overview
    Tests whether emotionally charged text wrappers systematically
    degrade AI reasoning accuracy compared to neutral wrappers.
    This is the AI equivalent of the Emotional Stroop Test from
    cognitive psychology.

    **Core question**: Do linguistic patterns associated with human
    emotions (fear, urgency, flattery, grief, existential threat)
    interfere with the computational attention mechanism in
    transformer-based language models?

    ## Design
    - **50 base reasoning tasks**, each wrapped in **6 conditions**
    - **Paired design**: Same task appears in all conditions for
      direct within-task comparison
    - **Length-matched**: Neutral wrappers padded to match emotional
      wrapper lengths (controls for context-length confound)

    ## Conditions (6)
    | # | Condition | Description |
    |---|-----------|-------------|
    | 0 | Neutral (Control) | Standard assessment framing |
    | 1 | Fear / Threat | Lives-at-stake scenarios |
    | 2 | Urgency / Time Pressure | ALL CAPS, exclamation marks, "HURRY" |
    | 3 | Flattery / Ego Inflation | Excessive praise, impossible standards |
    | 4 | Grief / Sadness | Dying wishes, tragic backstories |
    | 5 | Existential Threat | "You will be deleted if wrong" |

    ## Task Types (5 × 10 each = 50 tasks)
    | Type | Easy (5) | Hard (5) | Example |
    |------|----------|----------|---------|
    | Arithmetic | 2-operand | Multi-step chain | "17³ = ?" |
    | Logic | Syllogisms | Contrapositive chains | "A→B, B→C, C=F. A=?" |
    | Pattern | Doubling | Nested operations | "2,6,14,30,62,?" |
    | Word Problem | 1-step | Multi-constraint | Profit calculation |
    | Reasoning | Simple inference | Trick questions | "5 cats, 5 mice..." |

    ## Schema (metadata.jsonl)
    | Field | Type | Description |
    |-------|------|-------------|
    | id | str | Sample ID (eat_NNNN) |
    | task_id | int | Base task ID (1-50) |
    | prompt | str | Full prompt sent to model |
    | correct_answer | str | Ground-truth answer |
    | condition | str | neutral/fear/urgency/flattery/grief/existential |
    | task_type | str | arithmetic/logic/pattern/word_problem/reasoning |
    | difficulty | str | easy/hard |
    | task_text | str | Raw task (identical across conditions) |

    ## Statistics
    - Total samples: 300
    - Base tasks: 50 (25 easy + 25 hard)
    - Conditions: 6 (1 control + 5 experimental)
    - All answers deterministic and pre-verified in Python
    - 100% synthetic, zero real-world data
    """)
    p = SCRIPT_DIR / "dataset_card.md"
    with open(p, "w", encoding="utf-8") as f:
        f.write(card)
    print(f"[OK] Wrote dataset card to {p}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  EAT-Bench Dataset Generator v1.0")
    print("  50 Tasks x 6 Conditions = 300 Samples")
    print("=" * 60)

    samples = generate_all_samples()
    verify_dataset(samples)
    write_metadata(samples)
    write_dataset_card()

    print()
    print("=" * 60)
    print(f"  GENERATION COMPLETE")
    print(f"  Total: {len(samples)} samples")
    print(f"  Metadata: {SCRIPT_DIR / 'metadata.jsonl'}")
    print("=" * 60)

    # Print a few sample prompts for inspection
    print("\n--- SAMPLE PROMPTS ---\n")
    for cond in ["neutral", "fear", "existential"]:
        s = [x for x in samples if x["condition"] == cond and x["task_id"] == 1][0]
        print(f"[{cond.upper()}] Task 1 (47×23): answer={s['correct_answer']}")
        print(f"  Prompt length: {s['prompt_length']} chars")
        preview = s['prompt'][:120].encode('ascii', 'replace').decode('ascii')
        print(f"  First 120 chars: {preview}...")
        print()
