// ignore_for_file: depend_on_referenced_packages

import 'package:flutter/material.dart';
import 'package:mysql1/mysql1.dart' show ConnectionSettings, MySqlConnection;
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LawyerAI',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Colors.grey[900],
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.black87,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.black87,
            foregroundColor: Colors.white,
          ),
        ),
        floatingActionButtonTheme: FloatingActionButtonThemeData(
          backgroundColor: Colors.black87,
        ),
      ),
      home: const LoginPage(),
    );
  }
}

// Pagina de login
class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isPasswordVisible = false;

  // Conectarea la baza de date MySQL pentru autentificare
  Future<void> _login() async {
    String username = _usernameController.text;
    String password = _passwordController.text;

    if (username.isEmpty || password.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Introduceți utilizatorul și parola')),
      );
      return;
    }

    try {
      var settings = ConnectionSettings(
        host: 'localhost',
        port: 3306,
        user: 'root', 
        password: 'Titi300903!', 
        db: 'lawyerAI_users', 
      );

      var conn = await MySqlConnection.connect(settings);

      var results = await conn.query(
        'SELECT * FROM users WHERE username = ? AND password = ?',
        [username, password],
      );

      if (results.isNotEmpty) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => ChatPage(username: username, title: '',),
          ),
        );
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Autentificare eșuată!')),
        );
      }

      await conn.close();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Eroare de conectare la baza de date: $e')),
      );
    }
  }

  void _goToSignUp() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const SignUpPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('LawyerAI Login'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _usernameController,
              decoration: const InputDecoration(
                labelText: 'Utilizator',
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _passwordController,
              obscureText: !_isPasswordVisible,
              decoration: InputDecoration(
                labelText: 'Parola',
                border: const OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
                suffixIcon: IconButton(
                  icon: Icon(_isPasswordVisible ? Icons.visibility : Icons.visibility_off),
                  onPressed: () {
                    setState(() {
                      _isPasswordVisible = !_isPasswordVisible;
                    });
                  },
                ),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _login,
              child: const Text('Autentificare'),
            ),
            const SizedBox(height: 10),
            TextButton(
              onPressed: _goToSignUp,
              child: const Text('Nu ai cont? Înregistrează-te', style: TextStyle(color: Colors.white)),
            ),
          ],
        ),
      ),
    );
  }
}

// Pagina de înregistrare
class SignUpPage extends StatefulWidget {
  const SignUpPage({super.key});

  @override
  State<SignUpPage> createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  final _usernameController = TextEditingController();
  final _firstNameController = TextEditingController();
  final _lastNameController = TextEditingController();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  bool _isPasswordVisible = false;

  Future<void> _signUp() async {
    String username = _usernameController.text;
    String firstName = _firstNameController.text;
    String lastName = _lastNameController.text;
    String email = _emailController.text;
    String password = _passwordController.text;
    String confirmPassword = _confirmPasswordController.text;

    if (username.isEmpty || firstName.isEmpty || lastName.isEmpty || email.isEmpty || password.isEmpty || confirmPassword.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Completează toate câmpurile')),
      );
      return;
    }

    if (password != confirmPassword) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Parolele nu se potrivesc')),
      );
      return;
    }

    try {
      var settings = ConnectionSettings(
        host: 'localhost',
        port: 3306,
        user: 'root',
        password: 'Titi300903!',
        db: 'lawyerAI_users',
      );

      var conn = await MySqlConnection.connect(settings);

      await conn.query(
        'INSERT INTO users (username, first_name, last_name, email, password, confirm_password) VALUES (?, ?, ?, ?, ?, ?)',
        [username, firstName, lastName, email, password, confirmPassword],
      );

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Cont creat cu succes! Te poți autentifica.')),
      );

      await conn.close();
      Navigator.pop(context);
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Eroare la crearea contului: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('LawyerAI - Creează Cont'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _firstNameController,
              decoration: const InputDecoration(
                labelText: 'Prenume',
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _lastNameController,
              decoration: const InputDecoration(
                labelText: 'Nume',
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _usernameController,
              decoration: const InputDecoration(
                labelText: 'Utilizator',
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _emailController,
              decoration: const InputDecoration(
                labelText: 'Email',
                border: OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _passwordController,
              obscureText: !_isPasswordVisible,
              decoration: InputDecoration(
                labelText: 'Parola',
                border: const OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
                suffixIcon: IconButton(
                  icon: Icon(_isPasswordVisible ? Icons.visibility : Icons.visibility_off),
                  onPressed: () {
                    setState(() {
                      _isPasswordVisible = !_isPasswordVisible;
                    });
                  },
                ),
              ),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _confirmPasswordController,
              obscureText: !_isPasswordVisible,
              decoration: InputDecoration(
                labelText: 'Confirmă Parola',
                border: const OutlineInputBorder(),
                filled: true,
                fillColor: Colors.white10,
                suffixIcon: IconButton(
                  icon: Icon(_isPasswordVisible ? Icons.visibility : Icons.visibility_off),
                  onPressed: () {
                    setState(() {
                      _isPasswordVisible = !_isPasswordVisible;
                    });
                  },
                ),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _signUp,
              child: const Text('Înregistrează-te'),
            ),
          ],
        ),
      ),
    );
  }
}

// Pagina de chat
class ChatPage extends StatefulWidget {
  const ChatPage({super.key, required this.title, required this.username});

  final String title;
  final String username;

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final List<Map<String, String>> _messages = [];
  final _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<Color> _themeColors = [Colors.black, Colors.blueGrey, Colors.deepPurple];
  int _currentColorIndex = 0;

  // Metodă pentru a obține predicția de la server
  Future<void> _getPrediction(String message) async {
    try {
      final response = await http.post(
        Uri.parse('http://localhost:1234/predict'), // Adresa serverului Flask
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'message': message}),
      );

      if (response.statusCode == 200) {
        var result = jsonDecode(response.body);
        int predictedTag = result['predicted_tag'];
        String aiResponse;

        // Definește răspunsuri bazate pe clasa prezisă
        switch (predictedTag) {
          case 0:
            aiResponse = "Aceasta pare a fi o întrebare generală. Vă pot ajuta cu mai multe detalii?";
            break;
          case 1:
            aiResponse = "Aceasta pare a fi o încălcare a legii. Vă recomand să consultați un avocat.";
            break;
          case 2:
            aiResponse = "Aceasta este o cerere de clarificare juridică. Vă pot oferi informații generale.";
            break;
          default:
            aiResponse = "Nu am înțeles cererea. Vă rog să reformulați.";
        }

        setState(() {
          _messages.add({'sender': 'AI', 'text': aiResponse});
        });
      } else {
        setState(() {
          _messages.add({'sender': 'AI', 'text': 'Eroare la conectarea cu serverul.'});
        });
      }
    } catch (e) {
      setState(() {
        _messages.add({'sender': 'AI', 'text': 'Eroare: $e'});
      });
    }

    // Derulează la ultimul mesaj
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollController.animateTo(
        _scrollController.position.maxScrollExtent,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeOut,
      );
    });
  }

  void _sendMessage() {
    if (_messageController.text.isNotEmpty) {
      setState(() {
        _messages.add({'sender': 'You', 'text': _messageController.text});
      });

      // Trimite mesajul către server pentru predicție
      _getPrediction(_messageController.text);

      _messageController.clear();
    }
  }

  void _changeThemeColor() {
    setState(() {
      _currentColorIndex = (_currentColorIndex + 1) % _themeColors.length;
    });
    // Nu mai este necesară apelarea către _ChatPageState, deoarece setState actualizează UI-ul
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title.isEmpty ? 'LawyerAI Chat' : widget.title),
        actions: [
          IconButton(
            icon: const Icon(Icons.color_lens),
            onPressed: _changeThemeColor,
            tooltip: 'Schimbă Culoarea Temă',
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(8.0),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                final isUser = message['sender'] == 'You';
                return Align(
                  alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Container(
                    margin: const EdgeInsets.symmetric(vertical: 4.0),
                    padding: const EdgeInsets.all(12.0),
                    decoration: BoxDecoration(
                      color: isUser ? Colors.grey[700] : Colors.blueGrey[800],
                      borderRadius: BorderRadius.circular(12.0),
                    ),
                    child: Text(
                      '${message['text']}',
                      style: const TextStyle(color: Colors.white),
                    ),
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _messageController,
                    style: const TextStyle(color: Colors.white),
                    decoration: InputDecoration(
                      hintText: 'Întreabă-mă ceva...',
                      hintStyle: const TextStyle(color: Colors.grey),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12.0),
                      ),
                      filled: true,
                      fillColor: Colors.white10,
                    ),
                    onSubmitted: (_) => _sendMessage(),
                  ),
                ),
                const SizedBox(width: 8),
                FloatingActionButton(
                  onPressed: _sendMessage,
                  tooltip: 'Trimite',
                  child: const Icon(Icons.send),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}