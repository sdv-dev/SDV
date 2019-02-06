"""
Multiparent modelling example.

Altought it's unsupported yet, this example contains data to illustrate the behavior of SDV
when modelling and sampling multiparent tables, that is, tables that contain multiple foreign_keys
from other tables.

You can run this example with:

```
cd examples/multiparent_example
python multiparent_example.py
```

"""

from sdv import SDV


def run_example():
    """Example of usage of SDV for tables contanining more than one foreign key."""
    # Setup
    vault = SDV('data/meta.json')
    vault.fit()

    # Run
    result = vault.sample_all()

    for name, table in result.items():
        print('Samples generated for table {}:\n{}\n'.format(name, table.head(5)))


if __name__ == '__main__':
    run_example()
