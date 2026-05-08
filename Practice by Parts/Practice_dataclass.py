from dataclasses import dataclass

@dataclass(slots=True)
class StudentConfig:
    name:str = "Alice"
    age:int = 20
    major:str = "Statistics"
    gpa:float = 3.8

    def validate(self) -> None:
        if self.age <= 0:
            raise ValueError("How old are you????")
        if self.gpa < 0. or self.gpa > 4.0:
            raise ValueError("GPA must be between 0 and 4. You must take your undergraduate study on MARS.")

def print_student(config:StudentConfig) -> None:
        config.validate()
        print("Student Information:")
        print("---------------------")
        print("Name:\t",config.name , "\n",
              "Age:\t",config.age , "\n",
              "Major:\t",config.major , "\n",
              "GPA:\t",config.gpa 
              )

if __name__ == "__main__":
    default_student = StudentConfig()
    print("Default Student Information:")
    print_student(default_student)

    Pascal_Zhang = StudentConfig(name = "Pascal Zhang",
                                 age = 26,
                                 gpa = 3.8,
                                 major = "Statistics" )


    print("Pascal Zhang Student Information:")
    print_student(Pascal_Zhang)

    Invalid_student = StudentConfig(name = "Invalid Student",
                                    age = 24,
                                    gpa = 5.0,
                                    major = "Statistics" )
    try:
       print_student(Invalid_student)
    except ValueError as e:
        print("Caught ValueError: {}".format(e))

    try:
        default_student.gap = 4.0
    except AttributeError as e:
        print("Caught AttributeError: {}".format(e))

