# Calculator

> Perform basic math operations with internal memory

Calculator methods:

-   _add_ - add a real number to the internal memory value
-   _subtract_ - subtract a real number from the internal memory value
-   _multiply_ - multiply the internal memory value by a real number
-   _divide_ - divide the internal memory value by a real number
-   _root_ - take (n) root of the internal memory value
-   _reset_ - reset the internal memory value

## Installation

```sh
pip install pycalc3
```

## Usage

```python
>>> from pycalc3.calculator import Calculator
>>> cal = Calculator()
>>> cal.add(12)
>>> cal.memory_value
12.0
>>> cal.subtract(3)
>>> cal.memory_value
9.0
>>> cal.multiply(3)
>>> cal.memory_value
27.0
>>> cal.divide(3)
>>> cal.memory_value
9.0
>>> cal.root(2)
>>> cal.memory_value
3.0
>>> cal.reset()
>>> cal.memory_value
0.0
```

## [Changelog](https://github.com/simsiru/py-calculator/CHANGELOG.md)

## License

[MIT](https://choosealicense.com/licenses/mit/)