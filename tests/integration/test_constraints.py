from sdv.constraints import ColumnFormula, FixedCombinations, GreaterThan
from sdv.demo import load_tabular_demo
from sdv.tabular import GaussianCopula


def years_in_the_company(data):
    return data['age'] - data['age_when_joined']


def test_constraints(tmpdir):

    employees = load_tabular_demo()

    fixed_company_department_constraint = FixedCombinations(
        column_names=['company', 'department'],
        handling_strategy='transform',
        fit_columns_model=False
    )

    age_gt_age_when_joined_constraint = GreaterThan(
        low='age_when_joined',
        high='age',
        handling_strategy='reject_sampling',
        fit_columns_model=False
    )

    years_in_the_company_constraint = ColumnFormula(
        column='years_in_the_company',
        formula=years_in_the_company,
        handling_strategy='transform'
    )

    constraints = [
        fixed_company_department_constraint,
        age_gt_age_when_joined_constraint,
        years_in_the_company_constraint
    ]
    gc = GaussianCopula(constraints=constraints)
    gc.fit(employees)
    gc.save(tmpdir / 'test.pkl')
    gc = gc.load(tmpdir / 'test.pkl')
    gc.sample(10)
