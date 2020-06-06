from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(

    # Only sentence
    Biconditional(AKnight, And(AKnight, AKnave)),
    Biconditional(AKnave, Not(And(AKnave, AKnight)))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(

    # First sentence
    Biconditional(AKnight, And(AKnave, BKnave)),
    Biconditional(AKnave, Not(And(AKnave, BKnave))),

    # Second sentence
    Biconditional(BKnight, Not(And(AKnave, BKnave))),
    Biconditional(BKnave, And(AKnave, BKnave))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(

    # First sentence
    Biconditional(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Biconditional(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),

    # Second sentece
    Biconditional(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Biconditional(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight))))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."

knowledge3 = And(

    # First sentence
    Biconditional(AKnight, Or(AKnight, AKnave)),
    Biconditional(AKnave, Not(Or(AKnight, AKnave))),

    # Second sentence
    Biconditional(BKnight, Biconditional(AKnight, AKnave)),
    Biconditional(BKnight, Biconditional(AKnave, Not(AKnave))),

    Biconditional(BKnave, Not(Biconditional(AKnight, AKnave))),
    Biconditional(BKnave, Not(Biconditional(AKnave, Not(AKnave)))),

    # Third sentence
    Biconditional(BKnight, CKnave),
    Biconditional(BKnave, Not(CKnave)),

    # Fourth sentence
    Biconditional(CKnight, AKnight),
    Biconditional(CKnave, Not(AKnight))
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
