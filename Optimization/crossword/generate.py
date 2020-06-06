import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for key in list(self.domains):
            toRemove = []

            for value in self.domains[key]:
                if len(value) != key.length:
                    toRemove.append(value)

            for value in toRemove:
                self.domains[key].remove(value)

            # print(f'var: {key}, dom: {self.domains[key]}')

    def neighborhood(self, var):
        """
        Returns a list of pairs (x, y), where x is var and y is its neighboor.
        """
        queue = []
        for neighbor in self.crossword.neighbors(var):
            queue.append((var, neighbor))
        return queue

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        toDelete = []
        revised = False

        for value in self.domains[x]:
            satisfies = False
            w, z = self.crossword.overlaps[x, y]

            for corresp_value in self.domains[y]:
                if value != corresp_value:
                    if value[w] == corresp_value[z]:
                        satisfies = True
                        break

            if not satisfies:
                toDelete.append(value)
                revised = True

        if revised:
            for value in toDelete:
                self.domains[x].remove(value)

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = []

        if arcs is None:
            for var in self.crossword.variables:
                queue += self.neighborhood(var)
        else:
            queue = arcs

        while queue:
            x, y = queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    queue.append((x, z))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in self.crossword.variables:
            if var not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for var in list(assignment):
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    if assignment[var] == assignment[neighbor]:
                        return False
                    i, j = self.crossword.overlaps[var, neighbor]
                    if assignment[var][i] != assignment[neighbor][j]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        varDomain = {word: 0 for word in self.domains[var]}
        new_domain = []

        for word in list(varDomain):
            remains = self.crossword.neighbors(var) - set(assignment)

            for neighbor in remains:
                x, y = self.crossword.overlaps[var, neighbor]

                for value in self.domains[neighbor]:
                    if value[y] != word[x]:
                        varDomain[word] += 1

        domain = sorted(varDomain.items(), key=lambda x: x[1])
        for key in list(domain):
            new_domain.append(key[0])
        return new_domain

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        remains = self.crossword.variables - set(assignment)
        better = [None, 500, -1]  # variavel, domain_len, neighbor_len

        for var in remains:
            domain_len = len(self.domains[var])
            neighAvailable = len(self.crossword.neighbors(var)
                                 - set(assignment))

            if better[0] is None or domain_len < better[1]:
                better = [var, domain_len, neighAvailable]

            elif domain_len == better[1]:
                if neighAvailable > better[2]:
                    better = [var, domain_len, neighAvailable]

        return better[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        domain = self.order_domain_values(var, assignment)

        for value in domain:
            new_assignment = assigment.copy()
            new_assignment[var] = value

            if self.consistent(assignment):
                track = self.domains[var]
                neighboors = self.neighborhood(var)
                if self.ac3(neighboors):
                    result = self.backtrack(assignment)
                    if result:
                        return result
                self.domains[var] = track
        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
