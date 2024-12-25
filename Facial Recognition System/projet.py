import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os

# Chemins des ressources
project_path = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(project_path, "feature_extractor.tflite")
feature_db_path = os.path.join(project_path, "feature_db.npy")
icon_path = os.path.join(project_path, "reconnaissance-faciale.png")

# Initialisation TFLite et base de données
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
feature_db = np.load(feature_db_path, allow_pickle=True).item()

# Détecteur de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Variables pour la caméra
camera_running = False
cap = None

# Prétraitement de l'image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

# Extraction de caractéristiques
def extract_features_tflite(img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

# Identification de la personne
def identify_person(img_array, feature_db, threshold=0.9):
    query_feature = extract_features_tflite(img_array)
    for person, feature in feature_db.items():
        similarity = 1 - cosine(query_feature, feature)
        if similarity > threshold:
            return person, similarity * 100
    return "Inconnu", None

# Graphique de comparaison
def show_comparison_graph(input_features, matched_features, matched_person):
    plt.figure(figsize=(10, 6))
    plt.plot(input_features, label="Input Characteristics", color="blue", linestyle='-', marker='o', markersize=5, linewidth=2)
    if matched_features is not None:
        plt.plot(matched_features, label=f"Characteristics {matched_person}", color="orange", linestyle='-', marker='x', markersize=5, linewidth=2)
    else:
        plt.plot([], label="No match found", color="red", linestyle='-', marker='x', markersize=5, linewidth=2)
    plt.title("Comparison of input and output characteristics")
    plt.xlabel("Index of characteristics")
    plt.ylabel("Value of characteristics")
    plt.legend()
    plt.grid(True)
    plt.show()

# Gestion de la caméra
def start_camera():
    global camera_running, cap
    if camera_running:
        return
    camera_running = True
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera.")
        camera_running = False
        return

    camera_label.pack(pady=20)

    def process_frame():
        global camera_running
        while camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img_resized = cv2.resize(face_img_rgb, (224, 224))
                face_img_array = np.expand_dims(face_img_resized / 255.0, axis=0).astype(np.float32)
                person, similarity = identify_person(face_img_array, feature_db)
                color = (0, 255, 0) if similarity and similarity > 90 else (0, 0, 255)
                label = f"{person} ({similarity:.2f}%)" if similarity else "Inconnu"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((500, 400))
            img_tk = ImageTk.PhotoImage(img)
            camera_label.config(image=img_tk)
            camera_label.image = img_tk

        cap.release()
        camera_label.config(image=None)
    threading.Thread(target=process_frame, daemon=True).start()

def stop_camera():
    global camera_running
    if camera_running:
        camera_running = False
        cap.release()
        camera_label.pack_forget()

def test_image():
    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    if file_path:
        img_array = preprocess_image(file_path)
        person, similarity = identify_person(img_array, feature_db)
        input_features = extract_features_tflite(img_array)
        matched_features = feature_db.get(person) if person != "Unknown" else None

        if similarity:
            result_text.set(f"Identity: {person} ({similarity:.2f}%)")
        else:
            result_text.set("Identity: Unknown")

        show_comparison_graph(input_features, matched_features, person if similarity else "No match")

def show_team():
    team_members = "Development by::\n- bouagal houssem eddine"
    messagebox.showinfo("Development by", team_members)

# Interface de login
def login():
    username = username_entry.get()
    password = password_entry.get()

    # Ici on vérifie les identifiants
    if username == "admin" and password == "1234":
        messagebox.showinfo("Success", "Connection successful. Welcome Admin!")
        login_window.destroy()  # Fermer la fenêtre de login
        show_main_interface()  # Afficher l'interface principale
    else:
        messagebox.showerror("Error", "Incorrect username or password.")
def show_main_interface():
    # Interface principale
    global camera_label, result_text, root
    root = tk.Tk()
    root.title("Facial Recognition System")
    root.geometry("600x600")
    root.configure(bg="#f7f7f7")
    icon_image = tk.PhotoImage(file=icon_path)
    root.iconphoto(False, icon_image)

    header_label = tk.Label(root, text="Facial Recognition System", font=("Arial", 18, "bold"), bg="#007BFF", fg="white", pady=10)
    header_label.pack(fill=tk.X)

    test_button = tk.Button(root, text="Test an Image", font=("Arial", 12), bg="#007BFF", fg="white", command=test_image)
    test_button.pack(pady=10)

    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text, font=("Arial", 14), bg="#f7f7f7", fg="#333")
    result_label.pack(pady=10)

    camera_label = tk.Label(root, bg="#f7f7f7")

    button_frame = tk.Frame(root, bg="#f7f7f7")
    button_frame.pack(pady=10)

    button_width = 20
    camera_button = tk.Button(button_frame, text="Start the Camera", font=("Arial", 12), bg="#28A745", fg="white", command=start_camera, width=button_width)
    camera_button.grid(row=0, column=0, padx=5)

    stop_button = tk.Button(button_frame, text="Stop the Camera", font=("Arial", 12), bg="#DC3545", fg="white", command=stop_camera, width=button_width)
    stop_button.grid(row=0, column=1, padx=5)

    team_button = tk.Button(button_frame, text="Dev by", font=("Arial", 12), bg="#FFC107", fg="black", command=show_team, width=button_width)
    team_button.grid(row=0, column=2, padx=5)

    logout_button = tk.Button(root, text="Log out", font=("Arial", 12), bg="#FF5733", fg="white", command=logout, width=20)
    logout_button.pack(pady=20)

    root.protocol("WM_DELETE_WINDOW", lambda: (stop_camera(), root.destroy()))
    root.mainloop()

def logout():
    # Fermer la fenêtre principale
    root.destroy()
    # Réouvrir la fenêtre de login
    open_login_window()

def open_login_window():
    global login_window, username_entry, password_entry

    login_window = tk.Tk()
    login_window.title("Facial Recognition System")
    login_window.geometry("400x500")
    login_window.configure(bg="#f7f7f7")

    # Détecter automatiquement le chemin du fichier icône PNG
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reconnaissance-faciale.png")

    # Vérifier si le fichier existe
    if os.path.exists(icon_path):
        login_icon = tk.PhotoImage(file=icon_path)
        login_window.iconphoto(False, login_icon)
    else:
         print(f"Error: The icon '{icon_path}' was not found.")

    # Titre principal
    login_label = tk.Label(
        login_window, text="Facial Recognition System",
        font=("Arial", 18, "bold"), bg="#007BFF", fg="white", pady=10
    )
    login_label.pack(fill=tk.X)

    # Chargement et affichage de l'image
    if os.path.exists(icon_path):
        # Ajuster حجم الصورة ليكون أصغر قليلاً
        image = Image.open(icon_path).resize((120, 120), Image.LANCZOS)  # تعديل الحجم
        photo = ImageTk.PhotoImage(image)
        image_label = tk.Label(login_window, image=photo, bg="#f7f7f7")
        image_label.photo = photo  # Prévenir le garbage collector
        image_label.pack(pady=10)
    else:
          print(f"Error: The image '{icon_path}' was not found.")

    # Cadre pour les champs
    fields_frame = tk.Frame(login_window, bg="#f7f7f7")
    fields_frame.pack(pady=20)

    # Champ Nom d'utilisateur
    username_label = tk.Label(fields_frame, text="Username:", font=("Calibri", 12), bg="#f7f7f7")
    username_label.grid(row=0, column=0, padx=10, pady=5)
    username_entry = tk.Entry(fields_frame, font=("Calibri", 12))
    username_entry.grid(row=0, column=1, padx=10, pady=5)

    # Champ Mot de passe
    password_label = tk.Label(fields_frame, text="Password:", font=("Calibri", 12), bg="#f7f7f7")
    password_label.grid(row=1, column=0, padx=10, pady=5)
    password_entry = tk.Entry(fields_frame, show="*", font=("Calibri", 12))
    password_entry.grid(row=1, column=1, padx=10, pady=5)

    # Checkbox pour afficher le mot de passe
    show_password_var = tk.BooleanVar()
    show_password_checkbox = tk.Checkbutton(
        login_window, text="Show password", font=("Arial", 10),
        bg="#f7f7f7", variable=show_password_var,
        command=lambda: password_entry.config(show="" if show_password_var.get() else "*")
    )
    show_password_checkbox.pack(pady=10)

    # Bouton Se Connecter
    login_button = tk.Button(
        login_window, text="Log in", font=("Arial", 12),
        bg="#007BFF", fg="white", command=login
    )
    login_button.pack(pady=20)

    login_window.mainloop()

open_login_window()

