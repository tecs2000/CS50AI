import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )

        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probabilities = 1

    for name in people:
        p = 0
        # gene == 0
        if (name not in one_gene) and (name not in two_genes):
            p = PROBS['gene'][0]
            p *= PROBS['trait'][0][name in have_trait]

        # gene == 1 or gene == 2
        else:
            mother = {}
            father = {}

            # if is a mother or a father
            if people[name]['mother'] is None:
                if name in one_gene:
                    p = PROBS['gene'][1]
                    p *= PROBS['trait'][1][name in have_trait]

                else:
                    p = PROBS['gene'][2]
                    p *= PROBS['trait'][2][name in have_trait]

            # if is a son or a daughter
            else:
                mae = people[name]['mother']
                pai = people[name]['father']

                # mother has 0 copies of the gene
                if (mae not in one_gene) and (mae not in two_genes):
                    mother = {'yes': PROBS['mutation'],
                              'not': 1 - PROBS['mutation']}

                # mother has 1 copy of the gene
                elif mae in one_gene:
                    mother = {'yes': 0.5, 'not': 0.5}

                # mother has 2 copies of the gene
                else:
                    mother = {'yes': 1 - PROBS['mutation'],
                              'not': PROBS['mutation']}

                # father has 0 copies of the gene
                if (pai not in one_gene) and (pai not in two_genes):
                    father = {'yes': PROBS['mutation'],
                              'not': 1 - PROBS['mutation']}

                # father has 1 copy of the gene
                elif pai in one_gene:
                    father = {'yes': 0.5, 'not': 0.5}

                # father has 2 copies of the gene
                else:
                    father = {'yes': 1 - PROBS['mutation'],
                              'not': PROBS['mutation']}

                if name in one_gene:
                    # heritage from the mother OR from the father
                    p = mother['yes'] * father['not'] + \
                        mother['not'] * father['yes']

                    p *= PROBS['trait'][1][name in have_trait]

                else:
                    # heritage from the mother AND from the father
                    p = mother['yes'] * father['yes']
                    p *= PROBS['trait'][2][name in have_trait]

        probabilities *= p
    return probabilities


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in probabilities:
        if (name in one_gene) or (name in two_genes):
            if name in one_gene:
                probabilities[name]['gene'][1] += p
            else:
                probabilities[name]['gene'][2] += p

            probabilities[name]['trait'][name in have_trait] += p

        else:
            probabilities[name]['gene'][0] += p
            probabilities[name]['trait'][name in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        for treat_gene in probabilities[name]:
            soma = 0
            # sum all the values
            for key in probabilities[name][treat_gene]:
                soma += probabilities[name][treat_gene][key]

            factor = 1/soma
            for key in probabilities[name][treat_gene]:
                probabilities[name][treat_gene][key] *= factor


if __name__ == "__main__":
    main()
