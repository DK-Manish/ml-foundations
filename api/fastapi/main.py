from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello from Fast API"}


@app.get("/hello")
def say_hello():
    return {"message": "Hello Manish"}


@app.get("/search")
def search(q: str):
    return {"query": q}


class Student(BaseModel):
    id: int
    name: str
    marks: int


students = []


@app.post("/student")
def create_student(student: Student):
    students.append(student)
    return student


@app.get("/students")
def get_all_students():
    return students


@app.get("/students/{student_id}")
def get_student_by_id(student_id: int):
    for student in students:
        if student.id == student_id:
            return student
    return {"error": "Student not found"}


@app.put("/students/{student_id}")
def update_student(student_id: int, updated_student: Student):
    for i, student in enumerate(students):
        if student.id == student_id:
            students[i] = updated_student
            return updated_student
    return {"error": "Student not found"}


@app.delete("/students/{student_id}")
def delete_student(student_id: int):
    for i, student in enumerate(students):
        if student.id == student_id:
            deleted_student = students.pop(i)
            return {
                "message": "Student deleted successfully",
                "student": deleted_student
            }
    return {"error": "Student not found"}
