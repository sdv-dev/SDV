import warnings

from pandas.core.dtypes.common import is_object_dtype


class InvalidDataWarning(Warning):
    pass


def _process_adventure_works(data):
    data = data.copy()
    data['CountryRegion'].loc[data['CountryRegion']['Name'] == 'Namibia', 'CountryRegionCode'] = (
        'NA'
    )
    integer_comma_columns = {
        'PurchaseOrderDetail': [
            'ReceivedQty',
            'RejectedQty',
            'StockedQty',
        ],
        'BillOfMaterials': ['PerAssemblyQty'],
        'TransactionHistoryArchive': [],
    }
    float_comma_columns = {
        'Product': ['StandardCost', 'ListPrice', 'Weight'],
        'ProductListPriceHistory': ['ListPrice'],
        'ProductCostHistory': ['StandardCost'],
        'ProductVendor': ['StandardPrice', 'LastReceiptCost'],
        'SalesOrderDetail': ['UnitPrice', 'UnitPriceDiscount', 'LineTotal'],
        'SalesOrderHeader': ['SubTotal', 'TaxAmt', 'Freight', 'TotalDue'],
        'PurchaseOrderDetail': [
            'UnitPrice',
            'LineTotal',
        ],
        'PurchaseOrderHeader': ['SubTotal', 'TaxAmt', 'Freight', 'TotalDue'],
        'TransactionHistory': ['ActualCost'],
        'TransactionHistoryArchive': ['ActualCost'],
        'SalesTaxRate': ['TaxRate'],
        'EmployeePayHistory': ['Rate'],
        'ShipMethod': ['ShipBase', 'ShipRate'],
        'Location': ['CostRate', 'Availability'],
        'WorkOrderRouting': ['ActualResourceHrs', 'PlannedCost', 'ActualCost'],
        'SalesPersonQuotaHistory': ['SalesQuota'],
        'SalesPerson': [
            'SalesQuota',
            'Bonus',
            'CommissionPct',
            'SalesYTD',
            'SalesLastYear',
        ],
        'SalesTerritory': ['SalesYTD', 'SalesLastYear', 'CostYTD', 'CostLastYear'],
        'CurrencyRate': ['AverageRate', 'EndOfDayRate'],
        'SpecialOffer': ['DiscountPct'],
    }
    for table_name, columns in float_comma_columns.items():
        for column_name in columns:
            col = data[table_name][column_name]
            if is_object_dtype(col.dtype):
                data[table_name][column_name] = col.str.replace(',', '.').astype(float)

    for table_name, columns in integer_comma_columns.items():
        for column_name in columns:
            col = data[table_name][column_name]
            if is_object_dtype(col.dtype):
                data[table_name][column_name] = (
                    col.str.replace(',', '.').astype(float).astype('int64')
                )
    EMAIL_NAME_FIXES = {
        'fran?ois': 'francois',
        'jos?': 'jose',
        'j?sus': 'jesus',
        'mar?a': 'maria',
        'c?sar': 'cesar',
        'andr?s': 'andres',
        'ram?n': 'ramon',
        'bego?a': 'begona',
        'desir?e': 'desiree',
        'ren?e': 'renee',
        'al?cia': 'alicia',
        'ni?ia': 'ninia',
    }
    for table_name in ('EmailAddress', 'ProductReview'):
        emails = data[table_name]['EmailAddress']
        for broken, fixed in EMAIL_NAME_FIXES.items():
            emails = emails.str.replace(broken, fixed, regex=False)
        data[table_name]['EmailAddress'] = emails

    PHONE_COLUMNS = {
        'PersonPhone': ['PhoneNumber'],
    }

    for table_name, columns in PHONE_COLUMNS.items():
        for column_name in columns:
            col = data[table_name][column_name].astype('string')
            needs_prefix = col.notna() & ~col.str.startswith('+')
            data[table_name][column_name] = col.where(~needs_prefix, '+1 ' + col)

    return data


def process_adventure_works(data, metadata):
    try:
        metadata.validate_data(data)
        return data, metadata
    except Exception:
        try:
            processed_data = _process_adventure_works(data)
            metadata.validate_data(processed_data)
            return processed_data, metadata
        except Exception:
            warning_msg = (
                'This downloaded data and metadata are not compatible with each other. They '
                'cannot be modeled using SDV without some modifications.'
            )
            warnings.warn(warning_msg, InvalidDataWarning)
            return data, metadata
