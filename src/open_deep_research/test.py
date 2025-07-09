from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    sex: str
    height: float
    weight: float
michael = Person(name='Michael', age=15, sex='Male', height='1.81', weight='75')
jack: Person = {'name': 'Jack', 'age': "15", 'sex': 'Male', 'height': '1.81', 'weight': '75'}
print(michael)
print(jack)