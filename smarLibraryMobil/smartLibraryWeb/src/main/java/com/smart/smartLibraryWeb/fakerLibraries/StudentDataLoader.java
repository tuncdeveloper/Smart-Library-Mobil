//package com.smart.smartLibraryWeb.service.studentService;
//
//import com.smart.smartLibraryWeb.model.Book;
//import com.smart.smartLibraryWeb.model.Student;
//import com.smart.smartLibraryWeb.repository.StudentRepository;
//import jakarta.annotation.PostConstruct;
//import lombok.RequiredArgsConstructor;
//import org.springframework.stereotype.Component;
//
//import java.time.LocalDateTime;
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
//@Component
//@RequiredArgsConstructor
//public class StudentDataLoader {
//
//    private final StudentRepository studentRepository;
//
//
//
//
//    private static class StudentData {
//        String username;
//        String password;
//        String email;
//        String fullName;
//        String department;
//        Integer grade;
//        String phone;
//
//        public StudentData(String username, String password, String email, String fullName, String department, Integer grade, String phone) {
//            this.username = username;
//            this.password = password;
//            this.email = email;
//            this.fullName = fullName;
//            this.department = department;
//            this.grade = grade;
//            this.phone = phone;
//        }
//    }
//
//    private final List<StudentData> studentDataList = List.of(
//            new StudentData("aysegul", "abc123", "aysegul.kara@example.com", "Ayşegül Kara", "Elektrik-Elektronik Mühendisliği", 2, "0544 987 6543"),
//            new StudentData("elifd", "qwe456", "elif.demir@example.com", "Elif Demir", "Endüstri Mühendisliği", 1, "0507 111 2233"),
//            new StudentData("selint", "selin123", "selin.tuna@example.com", "Selin Tuna", "Mimarlık", 3, "0543 333 4455"),
//            new StudentData("zeynepa", "zey123", "zeynep.arslan@example.com", "Zeynep Arslan", "Psikoloji", 1, "0506 999 1122"),
//            new StudentData("nazlia", "naz456", "nazli.ay@example.com", "Nazlı Ay", "İşletme", 3, "0542 777 8899"),
//
//            // Yeni eklenen kız öğrenciler:
//            new StudentData("melisk", "melis123", "melis.korkmaz@example.com", "Melis Korkmaz", "Hukuk", 2, "0551 222 3344"),
//            new StudentData("cansuoz", "cansu789", "cansu.ozdemir@example.com", "Cansu Özdemir", "Tıp", 4, "0552 333 4455")
//    );
//
//
//    @PostConstruct
//    public void loadStudents() {
//        if (studentRepository.count() > 0) return;
//
//        List<Student> students = new ArrayList<>();
//
//        for (StudentData data : studentDataList) {
//            Student student = new Student();
//            student.setUsername(data.username);
//            student.setFullName(data.fullName);
//            student.setEmail(data.email);
//           student.setGrade(data.grade);
//           student.setDepartment(data.department);
//           student.setPassword(data.password);
//           student.setPhone(data.phone);
//            student.setCreatedAt(LocalDateTime.now());
//
//            students.add(student);
//        }
//
//
//        studentRepository.saveAll(students);
//        System.out.println(students.size() + " öğrenciler başarıyla yüklendi.");
//    }
//
//}
