Toy exercise: practice @dataclass from scratch

Goal:
Write this file from scratch to practice the meaning of @dataclass(slots=True).
Do not copy a solution first. Try to code the whole file yourself.

Requirements:
1. Import dataclass from dataclasses.
2. Define a dataclass named StudentConfig with slots=True.
3. StudentConfig should have the following fields:
   - name: str, default value "Alice"
   - age: int, default value 20
   - major: str, default value "Statistics"
   - gpa: float, default value 3.8
4. Inside StudentConfig, define a method validate(self) -> None.
5. validate should raise ValueError in the following cases:
   - age <= 0
   - gpa < 0 or gpa > 4
6. Define a function print_student(config: StudentConfig) -> None.
7. This function should first call config.validate().
8. Then it should print the student's information.
9. In the main block, create three examples:
   - One student using all default values.
   - One student overriding all values.
   - One invalid student with gpa=5.0, and catch the ValueError.
10. Also test slots=True by trying to assign a misspelled attribute:
    config.gap = 4.0
    Catch the AttributeError and print the error message.

Expected ideas to practice:
- @dataclass(slots=True)
- default field values
- type annotations
- instance methods
- ValueError
- AttributeError caused by slots=True
- if __name__ == "__main__":

Suggested expected output shape:
Student Information
-------------------
Name: Alice
Age: 20
Major: Statistics
GPA: 3.8

Student Information
-------------------
Name: Bob
Age: 22
Major: Mathematics
GPA: 3.6

Caught ValueError:
gpa must be between 0 and 4

Caught AttributeError:
...

Start coding below this line.