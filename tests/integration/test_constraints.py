from sdv.constraints import ColumnFormula, GreaterThan, UniqueCombinations
from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula


def years_in_the_company(data):
    return data['age'] - data['age_when_joined']


def test_constraints():

    employees = load_tabular_demo()

    unique_company_department_constraint = UniqueCombinations(
        columns=['company', 'department'],
        handling_strategy='transform'
    )

    age_gt_age_when_joined_constraint = GreaterThan(
        low='age_when_joined',
        high='age',
        handling_strategy='reject_sampling'
    )

    years_in_the_company_constraint = ColumnFormula(
        column='years_in_the_company',
        formula=years_in_the_company,
        handling_strategy='transform'
    )

    constraints = [
        unique_company_department_constraint,
        age_gt_age_when_joined_constraint,
        years_in_the_company_constraint
    ]
    gc = GaussianCopula(constraints=constraints)
    gc.fit(employees)
    gc.sample(10)
