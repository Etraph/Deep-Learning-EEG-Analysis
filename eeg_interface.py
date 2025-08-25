#---------------------------------Interface by Ethan Raphael and Mathis Letellier----------------------------------------------

# installer fpdf
# installer torch
# installer joblib

import customtkinter as ctk
import sqlite3

import tkinter as tk
from tkinter import messagebox, ttk, filedialog

#bibliothèques pour l'IA
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.signal import find_peaks
import scipy.signal
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le chemin de la database Bonn
malade_train_data_dir = "./Database Bonn/Train/Malade"
sain_train_data_dir = "./Database Bonn/Train/Sain"
malade_test_data_dir = "./Database Bonn/Test/Malade"
sain_test_data_dir = "./Database Bonn/Test/Sain"



# Configuration de base
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Connexion à la BDD SQLite créée jsute après
conn = sqlite3.connect('comptes_prat.db')
cursor = conn.cursor()

# Création de la table de comptes si non existante
cursor.execute('''CREATE TABLE IF NOT EXISTS comptes_prat (
    id_praticien TEXT PRIMARY KEY,
    nom TEXT,
    prenom TEXT,
    profession TEXT,
    laboratoire TEXT,
    mot_de_passe TEXT
)''')
conn.commit()

# Création de la table de comptes si non existante
cursor.execute('''CREATE TABLE IF NOT EXISTS Patients (
    id_patient TEXT PRIMARY KEY,
    id_praticien TEXT,
    nom TEXT,
    prenom TEXT,
    FOREIGN KEY(id_praticien) REFERENCES comptes_prat(id_praticien)
)''')
conn.commit()

cursor.execute('''CREATE TABLE IF NOT EXISTS Diagnostics (
    id_diag TEXT PRIMARY KEY,
    id_patient TEXT,
    donnees BLOB,
    resultat TEXT,
    FOREIGN KEY(id_patient) REFERENCES Patients(id_patient)
)''')
conn.commit()

RPPS_lst = [e[0] for e in cursor.execute('''SELECT * FROM comptes_prat''')]
SS_lst =  [e[0] for e in cursor.execute('''SELECT * FROM patients''')]


cursor.close()
class App(ctk.CTk):
    def __init__(self):
        def toggle_fullscreen(event=None): # ajout le 27/09/2024 (touche a attribuer encore)
              # Toggle entre le mode plein écran et normal
              state = not self.attributes('-fullscreen')                         # True si en fenêtré, False si en plein Ecran
              self.attributes('-fullscreen', state)                              # attribution du mode de fullscreen
        def confirm1(event=None):
            id_praticien, mot_de_passe = [entry.get() for entry in self.entries]
            cursor = conn.cursor()
            cursor.execute('''SELECT * FROM comptes_prat WHERE id_praticien = ? AND mot_de_passe = ?''', (id_praticien, mot_de_passe))
            account = cursor.fetchone()
            entry_mdp.delete(0, ctk.END)
            cursor.close()
            if account:
                entry_user.delete(0, ctk.END)
                self.frame_home.pack_forget()
                UserFrame(self, account).pack(fill="both", expand=True)
                self.unbind("<Return>") #evite que la touche entrée puisse encore etre utilisée
            else:
                messagebox.showerror("Erreur", "Identifiants incorrects")

        super().__init__()
        #self.attributes('-fullscreen', True)
        self.bind("<F11>", toggle_fullscreen)
        self.bind("<Return>",confirm1) #permet de confirmer la connexion au lieu de la touche entrée
        self.title("AI Diagnosis for EEG")
        largeur_fenetre = int(self.winfo_screenwidth())
        hauteur_fenetre = int(self.winfo_screenheight())
        self.geometry(f"{largeur_fenetre}x{hauteur_fenetre-290}-10-5") #-10-5 pour centrer la fenetre, c'est en pixel

        # Page d'accueil
        self.frame_home = ctk.CTkFrame(self)
        self.frame_home.pack(fill="both", expand=True)

        #self.logo = ctk.CTkLabel(self.frame_home, text="Logo")
        #self.logo.grid(row=0, column=0, padx=10, pady=10)

        #self.app_name = ctk.CTkLabel(self.frame_home, text="AI Diagnosis for EEG")
        #self.app_name.grid(row=0, column=1, padx=10, pady=10)



        self.home_content_frame = ctk.CTkFrame(self.frame_home, width=500, height=800, corner_radius=10)
        self.home_content_frame.place(relx=0.5, rely=0.5,anchor=tk.CENTER)

        self.title_label = ctk.CTkLabel(self.home_content_frame, text="Connexion à l'espace Praticiens", font=("Arial", 20))
        self.title_label.pack(pady=20)



        label_user = ctk.CTkLabel(self.home_content_frame, text="N° de praticien :")
        label_user.pack(padx=5)
        entry_user = ctk.CTkEntry(self.home_content_frame)
        entry_user.pack()

        label_mdp = ctk.CTkLabel(self.home_content_frame, text="Mot de passe :")
        label_mdp.pack(padx=5)
        entry_mdp = ctk.CTkEntry(self.home_content_frame, show="*")
        entry_mdp.pack()
        self.entries = [entry_user,entry_mdp]


        self.btn_confirm1 = ctk.CTkButton(self.home_content_frame, text="Se connecter", command=confirm1)
        self.btn_confirm1.pack(side="left",padx=10, pady=10)


        self.btn_login = ctk.CTkButton(self.home_content_frame, text="Créer un compte", command=self.open_create_account) # a modifier pour mettre directement la zone de login et en bouton connexion ou creation de compte
        self.btn_login.pack(side = "left", padx=10)

    def quitter(self):
        self.destroy()
        self.quit()


 #ouverture onglet de création de compte
    def open_create_account(self):
        self.frame_home.pack_forget()
        self.unbind("<Return>") #evite que la touche entrée puisse encore etre utilisée
        self.frame_create_account = CreateAccountFrame(self)

        self.frame_create_account.pack(fill="both", expand=True)
#ouverture de la page principale
    def open_home(self):
        if hasattr(self, 'frame_create_account'):
            self.frame_create_account.destroy()
        self.frame_home.pack(fill="both", expand=True)
#onglet de création de compte




class CreateAccountFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.title_label = ctk.CTkLabel(self, text="Créer un compte", font=("Arial", 20))
        self.title_label.pack(pady=20)



        self.frame_home = ctk.CTkFrame(self)
        self.frame_home.pack(fill="both", expand=True)

        self.logo = ctk.CTkLabel(self.frame_home, text="Logo")
        self.logo.grid(row=0, column=0, padx=10, pady=10)

        self.app_name = ctk.CTkLabel(self.frame_home, text="AI Diagnosis for EEG")
        self.app_name.grid(row=0, column=1, padx=10, pady=10)


        #self.bind("<Enter>",self.confirm2)

        self.home_content_frame = ctk.CTkFrame(self.frame_home, width=500, height=800, corner_radius=10)
        self.home_content_frame.place(relx=0.5, rely=0.5,anchor=tk.CENTER)

        self.entries = []
        for field in ["N° de praticien", "Nom", "Prénom", "Profession", "Laboratoire", "Mot de passe"]:
            frame = ctk.CTkFrame(self.home_content_frame)
            frame.pack(pady=5)
            label = ctk.CTkLabel(frame, text=f"{field} :")
            label.pack(side="left")
            entry = ctk.CTkEntry(frame)
            entry.pack(side="left", padx=10)
            self.entries.append(entry)

        self.btn_confirm2 = ctk.CTkButton(self.home_content_frame, text="Confirmer", command=self.confirm2)
        self.btn_confirm2.pack(pady=10)

        self.btn_back = ctk.CTkButton(self.home_content_frame, text="Retour", command=self.go_back)
        self.btn_back.pack(pady=10)
#bouton
    def confirm2(self,event=None):
        data = [entry.get() for entry in self.entries]
        try:
            int(data[0])
            if len(data[0]) != 11:
                messagebox.showerror("Erreur","Le RPPS est un nombre à 11 chiffres")

            elif data[0] in RPPS_lst:
                messagebox.showerror("Erreur","Le RPPS deja existant")
            elif not all(data):
                messagebox.showerror("Erreur", "Veuillez remplir tous les champs")
            else:
                cursor = conn.cursor()
                cursor.execute('''INSERT INTO comptes_prat (id_praticien, nom, prenom, profession, laboratoire, mot_de_passe)
                                  VALUES (?, ?, ?, ?, ?, ?)''', data)
                conn.commit()

                RPPS_lst.append(data[0])

                # Create personal databases for the new user
                messagebox.showinfo("Et voilà !", "Compte créé avec succès !")
                #self.master.open_home()
                cursor.execute('''SELECT * FROM comptes_prat WHERE id_praticien = ? AND mot_de_passe = ?''', (data[0], data[5]))
                account = cursor.fetchone()
                cursor.close()
                for e in self.entries:
                    e.delete(0,ctk.END)
                self.frame_home.pack_forget()
                UserFrame(self, account).pack(fill="both", expand=True)
                cursor.close()

        except:
            messagebox.showerror("Erreur","Le RPPS est un nombre")





    def go_back(self):
        self.pack_forget()
        self.master.frame_home.pack(fill="both", expand=True)



#le truc vraiment vraiment long
class UserFrame(ctk.CTkFrame):
    def __init__(self, master, user_info):
        super().__init__(master)
        self.user_info = user_info
        self.id_praticien = user_info[0]
        self.current_patient = None  # Variable pour stocker le patient actuellement sélectionné

        # Variables pour le réentraînement
        self.accepted_diagnoses_count = 0
        self.accepted_eeg_files = []
        self.accepted_labels = []

# grid pour lancer correctement tous les widgets en fonction des autres
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Top Bar
        #self.top_bar = ctk.CTkFrame(self, height=10, corner_radius=0)
        #self.top_bar.grid(row=0, columnspan=4, sticky="ew")

        #self.logo = ctk.CTkLabel(self.top_bar, text="Logo")
        #self.logo.grid(row=0, column=1, padx=10)

        #self.app_name = ctk.CTkLabel(self.top_bar, text="AI Diagnosis for EEG")
        #self.app_name.grid(row=0, column=2, padx=10)

        #grid pour top_bar
        #self.top_bar.grid_columnconfigure(0, weight=1)
        #self.top_bar.grid_columnconfigure(1, weight=1)
        #self.top_bar.grid_columnconfigure(2, weight=1)
        #self.top_bar.grid_rowconfigure(0, weight=0)

        #grid pour user_bar
        self.user_bar = ctk.CTkFrame(self, height=10, corner_radius=0)
        self.user_bar.grid(row=0, columnspan=3, pady=10, sticky="ew")

        #self.user_image = ctk.CTkLabel(self.user_bar, text="Image")
        #self.user_image.grid(row=0, column=0, padx=10)
        
        self.menu_button = ctk.CTkButton(self.user_bar, text="≡", width=30, command=self.open_menu)
        self.menu_button.grid(row=0, column=0, padx=10, sticky="w")

        self.new_patient_button = ctk.CTkButton(self.user_bar, text="Nouveau Patient", command=self.open_new_patient)
        self.new_patient_button.grid(row=0, column=1, pady=10)
        
        self.search_var = tk.StringVar()
        search_entry = ctk.CTkEntry(self.user_bar, textvariable=self.search_var, placeholder_text="Rechercher un patient...")
        search_entry.grid(row=0, column=2, padx=10)

        search_button = ctk.CTkButton(self.user_bar, text="Rechercher", command=self.search_patient)
        search_button.grid(row=0, column=3, padx=10)
        
        self.user_name = ctk.CTkLabel(self.user_bar, text=f"Dr. {user_info[1]}")
        self.user_name.grid(row=0, column=4, padx=10)

        self.user_profession = ctk.CTkLabel(self.user_bar, text=user_info[3])
        self.user_profession.grid(row=0, column=5, padx=10)

        self.user_laboratory = ctk.CTkLabel(self.user_bar, text=user_info[4])
        self.user_laboratory.grid(row=0, column=6, padx=10)


        self.user_bar.grid_rowconfigure(0, weight=0)
        self.user_bar.grid_columnconfigure(0, weight=1)
        self.user_bar.grid_columnconfigure(1, weight=1)
        self.user_bar.grid_columnconfigure(2, weight=1)
        self.user_bar.grid_columnconfigure(3, weight=1)
        self.user_bar.grid_columnconfigure(4, weight=1)
        self.user_bar.grid_columnconfigure(5, weight=1)
        self.user_bar.grid_columnconfigure(6, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        # Section gauche
        self.left_section = ctk.CTkFrame(self, width=200, height=400, fg_color=("blue", "purple"))
        self.left_section.grid(row=1, column=0, sticky="nswe", padx=10, pady=10)
        self.left_section.grid_rowconfigure(1, weight=1)
        self.left_section.grid_columnconfigure(0, weight=1)
        self.left_section.grid_rowconfigure(0, weight=0)

        self.left_section_title = ctk.CTkLabel(self.left_section, text="Patients", font=("Arial", 16))
        self.left_section_title.grid(row=0, column=0, pady=10)

        #Liste qui s'actualise
        self.patient_list_frame = ctk.CTkFrame(self.left_section)
        self.patient_list_frame.grid(row=2, column=0, sticky="nswe")
        self.patient_list_frame.grid_rowconfigure(0, weight=1)
        self.patient_list_frame.grid_columnconfigure(0, weight=1)

        # Section centrale
        self.center_section = ctk.CTkFrame(self, width=400, height=400, fg_color="white")
        self.center_section.grid(row=1, column=1, sticky="nswe", padx=10, pady=10)

        # Section droite
        self.right_section = ctk.CTkFrame(self, width=200, height=400, fg_color=("blue", "purple"))
        self.right_section.grid(row=1, column=2, sticky="nswe", padx=10, pady=10)

        self.load_patients()
#menu
    def open_menu(self):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Paramètres")
        menu.add_command(label="Déconnexion", command=self.logout)
        menu.add_command(label="Quitter", command=self.close)
        menu.tk_popup(self.winfo_rootx(), self.winfo_rooty())
    def close(self):
        self.master.destroy()
        self.master.quit()
    def logout(self):
        self.pack_forget()
        self.master.frame_home.pack(fill="both", expand=True)
#génère la liste des patients spécifiques au compte ouvert
    def load_patients(self):

        cursor = conn.cursor()
        for widget in self.patient_list_frame.winfo_children():
            widget.destroy()
        try:
            cursor.execute('''SELECT id_patient,id_praticien, nom, prenom FROM Patients''')
            patients = cursor.fetchall()
        except:
            messagebox.showerror("erreur","loadpatient error")

        for patient in patients:
            patient_button = ctk.CTkButton(self.patient_list_frame, text=f"{patient[2]} {patient[3]}",
                                           command=lambda p=patient: self.load_diagnostics(p))
            patient_button.pack(fill="x", pady=5)
        cursor.close()
#onglet d'ajout des nouveaux patients (nom, prénom, la clé est incrémentée toute seule)
    def open_new_patient(self):
        for widget in self.center_section.winfo_children():
            widget.destroy()

        self.new_patient_label = ctk.CTkLabel(self.center_section, text="Nouveau Patient", font=("Arial", 16))
        self.new_patient_label.pack(pady=10)

        self.new_patient_entries = []
        for field in ["N° Sécu Sociale","Nom", "Prénom"]:
            frame = ctk.CTkFrame(self.center_section)
            frame.pack(pady=5)
            label = ctk.CTkLabel(frame, text=f"{field} :")
            label.pack(side="left")
            entry = ctk.CTkEntry(frame)
            entry.pack(side="left", padx=10)
            self.new_patient_entries.append(entry)

        self.btn_add_patient = ctk.CTkButton(self.center_section, text="Ajouter", command=self.add_new_patient)
        self.btn_add_patient.pack(pady=10)

    def add_new_patient(self):
        cursor = conn.cursor()
        SS, nom, prenom = (entry.get() for entry in self.new_patient_entries)
        if len(SS)!=13:
            messagebox.showerror("erreur","le numéro de sécurité sociale est un nombre à 13 chiffres")
        elif nom and prenom:
            try:
                int(SS)
            except:
                messagebox.showerror("erreur","Le numéro de Sécurité sociale est invalide")
            cursor.execute('''INSERT INTO Patients (id_patient, id_praticien, nom, prenom)
                              VALUES (?, ?, ?, ?)''', (SS,self.user_info[0], nom, prenom))
            conn.commit()
            SS_lst.append(SS)
            self.load_patients()  # Rafraîchissement de la liste des patients
            for widget in self.center_section.winfo_children():
                widget.destroy()
               # messagebox.showerror("erreur","le n° sécurité sociale est déjà existant")
        else:
            messagebox.showerror("Erreur", "Veuillez remplir tous les champs")
        cursor.close()

#paramétrisation de la barre de recherche

    def search_patient(self):
        search_query = self.search_var.get().strip().lower()
        cursor = conn.cursor()
    
    #récupérer les patients correspondant au praticien connecté et au texte recherché
        query = '''
        SELECT id_patient, nom, prenom 
        FROM Patients 
        WHERE id_praticien = ? AND (LOWER(nom) LIKE ? OR LOWER(prenom) LIKE ?)
        '''
        search_term = f"%{search_query}%"  #permet la recherche partielle
        cursor.execute(query, (self.id_praticien, search_term, search_term))
        matching_patients = cursor.fetchall()
        cursor.close()
        
    #actualiser la liste des patients affichés
        for widget in self.patient_list_frame.winfo_children():
            widget.destroy()
            
        if matching_patients:
            for patient in matching_patients:
                patient_button = ctk.CTkButton(
                    self.patient_list_frame,
                    text=f"{patient[1]} {patient[2]}",
                    command=lambda p=patient: self.load_diagnostics(p)
                )
                patient_button.pack(fill="x", pady=5)
        else:
            no_patient_label = ctk.CTkLabel(self.patient_list_frame, text="Aucun patient trouvé.")
            no_patient_label.pack(pady=10)
        
#onglet de génération des diagnostics (section droite)
    def load_diagnostics(self, patient):
        cursor = conn.cursor()
        self.current_patient = patient  # Stockage des infos du patient sélectionné

        for widget in self.right_section.winfo_children():
            widget.destroy()
        cursor.execute('''SELECT resultat FROM Diagnostics WHERE id_patient = ?''', (patient[0],))
        diagnostics = cursor.fetchall()

        if diagnostics:
            for diag in diagnostics:
                diag_label = ctk.CTkLabel(self.right_section, text=diag[0])
                diag_label.pack(pady=5)
        else:
            no_diag_label = ctk.CTkLabel(self.right_section, text="Aucun diagnostic trouvé.")
            no_diag_label.pack(pady=10)

        # Ajout du bouton "Nouveau diagnostic" dans la section centrale
        for widget in self.center_section.winfo_children():
            widget.destroy()

        new_diag_button = ctk.CTkButton(self.center_section, text="Nouveau Diagnostic", command=self.open_new_diagnostic)
        new_diag_button.pack(pady=10)
        cursor.close()
#Nouveau diagnostic
#A MODIFIER : ENTREE SOUS FORMAT .txt, PAS UNE LISTE

    def open_new_diagnostic(self):

        for widget in self.center_section.winfo_children():
            widget.destroy()

        self.new_diag_label = ctk.CTkLabel(self.center_section, text="Nouveau Diagnostic", font=("Arial", 16))
        self.new_diag_label.pack(pady=10)

    # Bouton pour déposer un fichier
        self.file_path = None

        def open_file():
            # Interface pour sélectionner le fichier EEG
            root = tk.Tk()
            root.withdraw()  # Masquer la fenêtre principale

            file_path = filedialog.askopenfilename(title="Sélectionnez le fichier EEG", filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")])

            if file_path:
                self.file_path = file_path
                file_label.configure(text=f"Fichier sélectionné: {file_path}")

            # Activer le bouton de diagnostic
                diag_button.configure(state="normal")

        file_button = ctk.CTkButton(self.center_section, text="Déposer un fichier .txt", command=open_file)
        file_button.pack(pady=10)

        file_label = ctk.CTkLabel(self.center_section, text="Aucun fichier sélectionné")
        file_label.pack(pady=10)

    # Bouton pour lancer le diagnostic via l'IA
        def run_diagnostic():  # fichier qui lit le fichier txt
            if not self.file_path:
                messagebox.showerror("Erreur", "Veuillez d'abord sélectionner un fichier .txt")
            else:
                model, scaler, accuracy = load_model_and_scaler(model_path='eeg_model.pth', scaler_path='scaler.pkl')
                eeg_data = load_eeg_data(self.file_path)
                resultat_ia, prediction_label = diagnostic(eeg_data, model, scaler, device)
                self.prediction_label = prediction_label
                self.show_diagnosis_popup(resultat_ia, accuracy)

        diag_button = ctk.CTkButton(self.center_section, text="Lancer le diagnostic", command=run_diagnostic, state="disabled")
        diag_button.pack(pady=10)



# Popup pour confirmation/refus
    def show_diagnosis_popup(self, diagnostic, accuracy):
        popup = ctk.CTkToplevel(self)
        popup.title("Résultat du Diagnostic")
        popup.geometry("700x300")  # You can adjust the size as needed

        # Create a frame for content
        content_frame = ctk.CTkFrame(popup)
        content_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Display the diagnosis result
        result_label = ctk.CTkLabel(content_frame, text="Résultat du diagnostic IA", font=("Arial", 16))
        result_label.pack(pady=10)

        diagnostic_label = ctk.CTkLabel(content_frame, text=diagnostic, font=("Arial", 14))
        diagnostic_label.pack(pady=10)

        # Display the model accuracy if available
        if accuracy is not None:
            accuracy_label = ctk.CTkLabel(content_frame, text=f"Précision du modèle : {accuracy:.2f}%", font=("Arial", 12))
            accuracy_label.pack(pady=10)

        # Buttons frame
        buttons_frame = ctk.CTkFrame(content_frame)
        buttons_frame.pack(pady=10)

        def accept_diagnosis():
            self.save_diagnostic(diagnostic, "Accepté")
            # Incrémenter le compteur
            self.accepted_diagnoses_count += 1
            # Ajouter le fichier EEG et le label aux listes
            self.accepted_eeg_files.append(self.file_path)
            self.accepted_labels.append(self.prediction_label)
            # Vérifier si 10 diagnostics ont été acceptés
            if self.accepted_diagnoses_count >= 10:
                # Réentraîner le modèle
                model, scaler, _ = load_model_and_scaler(model_path='eeg_model.pth', scaler_path='scaler.pkl')
                model = retrain_model(model, scaler, self.accepted_eeg_files, self.accepted_labels, device)
                # Ajouter les nouveaux fichiers EEG dans la base de données
                copy_eeg_files(self.accepted_eeg_files, self.accepted_labels)
                # Sauvegarder le modèle mis à jour
                torch.save(model.state_dict(), 'eeg_model.pth')
                # Évaluer le modèle
                accuracy = evaluation_model(model, device, scaler)
                # Afficher la nouvelle précision
                messagebox.showinfo("Mise à jour du modèle", f"Le modèle a été mis à jour. Nouvelle précision: {accuracy:.2f}%")
                # Réinitialiser le compteur et les listes
                self.accepted_diagnoses_count = 0
                self.accepted_eeg_files = []
                self.accepted_labels = []
            popup.destroy()

        def reject_diagnosis():
            self.save_diagnostic(diagnostic, "Réfuté")
            popup.destroy()

        def view_curve():
            if self.file_path:
                eeg_data = load_eeg_data(self.file_path)
                plt.figure(figsize=(10, 4))
                plt.plot(eeg_data)
                plt.title("EEG du patient")
                plt.xlabel("Temps")
                plt.ylabel("Amplitude")
                plt.show()

        # Accept and Reject Buttons
        accept_button = ctk.CTkButton(buttons_frame, text="Accepter", command=accept_diagnosis, fg_color="green")
        accept_button.pack(side="left", padx=10)

        reject_button = ctk.CTkButton(buttons_frame, text="Réfuter", command=reject_diagnosis, fg_color="red")
        reject_button.pack(side="left", padx=10)

        view_curve_button = ctk.CTkButton(buttons_frame, text="Voir les courbes", command=view_curve)
        view_curve_button.pack(side="left", padx=10)


# Sauvegarde du diagnostic après confirmation
    def save_diagnostic(self, diagnostic, status):
        conn = sqlite3.connect('comptes_prat.db')
        cursor=conn.cursor()
        cursor.execute('''INSERT INTO Diagnostics (id_patient, donnees, resultat)
                          VALUES (?, ?, ?)''', (self.current_patient[0], self.file_path, f"{diagnostic} ({status})"))
        conn.commit()
        cursor.close()
        conn.close()
        self.load_diagnostics(self.current_patient) # Rafraîchissement de la liste des diagnostics

        self.generate_pdf_report(diagnostic, status)

    def generate_pdf_report(self, diagnostic, status):
        # Create a PDF object
        pdf = FPDF()
        pdf.add_page()

        # Set title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Rapport de Diagnostic", ln=1, align='C')
        pdf.cell(0,10,datetime.now().strftime('%d / %m / %Y à %H : %M'), ln=1, align ='C')

        # Practitioner Information
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 10, f"Praticien: Dr. {self.user_info[1]} {self.user_info[2]}", ln=1)
        pdf.cell(0, 10, f"Laboratoire: {self.user_info[4]}", ln=1)

        # Patient Information
        pdf.cell(0, 10, f"Patient: {self.current_patient[2]} {self.current_patient[3]}", ln=1)
        pdf.cell(0, 10, f"N° de Sécurité Sociale: {self.current_patient[0]}", ln=1)

        # AI Diagnosis Details
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Résultat du Diagnostic IA:", ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, f"{diagnostic} ({status})")

        # Include model accuracy
        # Load the model accuracy
        try:
            with open('model_accuracy.txt', 'r') as f:
                accuracy = float(f.read())
            pdf.cell(0, 10, f"Précision du modèle: {accuracy:.2f}%", ln=1)
        except FileNotFoundError:
            pdf.cell(0, 10, "Précision du modèle: N/A", ln=1)

        # Explanations and Database Info
        pdf.cell(0, 10, "Informations sur la base de données: Base de données Bonn EEG", ln=1)
        pdf.multi_cell(0, 10, "Le modèle a été entraîné sur la base de données EEG de Bonn, comprenant des enregistrements EEG de patients sains et épileptiques.")

        # Add EEG Graph
        if self.file_path:
            eeg_data = load_eeg_data(self.file_path)
            plt.figure(figsize=(10, 4))
            plt.plot(eeg_data)
            plt.title("EEG du patient")
            plt.xlabel("Temps")
            plt.ylabel("Amplitude")
            graph_path = "eeg_plot.png"
            plt.savefig(graph_path)
            plt.close()

            # Add the image to the PDF
            pdf.image(graph_path, x=10, y=None, w=pdf.w - 20)  # Adjust width as needed

            # Remove the temporary image file
            os.remove(graph_path)

        # Save the PDF with a unique filename
        report_filename = f"Diagnostic_{self.current_patient[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(report_filename)

        messagebox.showinfo("Rapport généré", f"Le rapport a été généré: {report_filename}")



#---------------------------------AI functions - work of Florian Mourey and Naïs Atlan -------------------------------------------------
# Extraction de features
def extract_features(eeg_data):
    sampling_rate = 173.61
    mean_feature = np.mean(eeg_data)
    max_feature = np.max(eeg_data)
    min_feature = np.min(eeg_data)
    std_feature = np.std(eeg_data)
    ptp_feature = np.ptp(eeg_data)
    skew_feature = stats.skew(eeg_data)
    var_feature = np.var(eeg_data)
    argmin_feature = np.argmin(eeg_data)
    argmax_feature = np.argmax(eeg_data)
    rms_feature = np.sqrt(np.abs(np.mean(eeg_data)))
    abs_diff_signal_feature = np.sum(np.abs(np.diff(eeg_data)))
    kurtosis_feature = stats.kurtosis(eeg_data)
    peaks, _ = find_peaks(eeg_data, height=mean_feature + std_feature/2)
    durations = len(np.diff(peaks))
    peaks_neg, _ = find_peaks(-np.array(eeg_data), height=mean_feature - std_feature/2)
    durations_neg = len(np.diff(peaks_neg))
    freqs, power = scipy.signal.welch(eeg_data, fs=sampling_rate)
    power_ratio_delta = power[(freqs >= 0.5) & (freqs <= 4)].sum() / power.sum()
    power_ratio_theta = power[(freqs > 4) & (freqs <= 8)].sum() / power.sum()

    return [mean_feature, std_feature, max_feature, min_feature, ptp_feature,
            skew_feature, var_feature, argmin_feature, argmax_feature, rms_feature,
            abs_diff_signal_feature, kurtosis_feature, durations, durations_neg,
            power_ratio_delta, power_ratio_theta]

# Modèle de neurones
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, 4)
        self.fc7 = nn.Linear(4, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.fc7(x)
        return x

#Fonction de diagnostic
def diagnostic(eeg_data, model, scaler, device):
    # Extraire les caractéristiques
    features = extract_features(eeg_data)
    features = np.array(features).reshape(1, -1)

    # Normaliser les caractéristiques avec le scaler chargé
    features_scaled = scaler.transform(features)

    # Convertir en tenseur PyTorch
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    # Désactiver la calcul des gradients
    with torch.no_grad():
        # Passer les données à travers le modèle
        outputs = model(features_tensor)
        # Obtenir la classe prédite
        _, predicted = torch.max(outputs, 1)
        prediction = predicted.item()

    # Interpréter la prédiction
    if prediction == 1:
        return "Épilepsie détectée.", prediction
    else:
        return "Épilepsie non détectée.", prediction


# Fonction de chargement des données
def load_data(data_dir):
  data = []
  label =[]
  for file_name in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'r') as file:
      lines = file.readlines()
      eeg_data = []
      for line in lines:
        eeg_data.append(int(line))
      features = extract_features(eeg_data)
      data.append(features)
      label.append(1 if (data_dir == malade_train_data_dir or data_dir == malade_test_data_dir) else 0)
  return np.array(data), np.array(label)

# Fonction pour charger les données EEG depuis un fichier
def load_eeg_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            eeg_data = [float(line.strip()) for line in lines]
        return eeg_data
    else:
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

# Fonction pour geler certaines couches d'un modèle
def freeze_model_layers(model, freeze_until_layer):
    """
    Cette fonction gèle toutes les couches d'un modèle jusqu'à la couche spécifiée.
    Seules les couches après 'freeze_until_layer' continueront à être entraînées.
    """
    freeze_flag = False

    for name, param in model.named_parameters():
        if name.startswith(freeze_until_layer):
            freeze_flag = True  # À partir de cette couche, on commence à dégeler

        if freeze_flag:
            param.requires_grad = True  # Les couches après freeze_until_layer seront entraînées
        else:
            param.requires_grad = False  # Les couches avant freeze_until_layer seront gelées



def retrain_model(model, scaler, eeg_data_list, labels_list, device, num_epochs=20, learning_rate=1e-7):
    # Préparer les caractéristiques
    features_list = []
    for eeg_data in eeg_data_list:
        features = extract_features(load_eeg_data(eeg_data))
        features_list.append(features)

    # Convertir en tableaux numpy
    X_new = np.array(features_list)
    y_new = np.array(labels_list)

    # Charger quelques anciennes données pour éviter l'oubli catastrophique
    malade_data, malade_labels = load_data(malade_train_data_dir)
    sain_data, sain_labels = load_data(sain_train_data_dir)
    X_combined = np.concatenate((X_new, malade_data, sain_data), axis=0)
    y_combined = np.concatenate((y_new, malade_labels, sain_labels), axis=0)

    # Normaliser les caractéristiques
    X_combined_scaled = scaler.transform(X_combined)

    # Convertir en tenseurs PyTorch
    X_combined_tensor = torch.tensor(X_combined_scaled, dtype=torch.float32).to(device)
    y_combined_tensor = torch.tensor(y_combined, dtype=torch.long).to(device)

    # Créer un DataLoader
    dataset = TensorDataset(X_combined_tensor, y_combined_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Définir la fonction de perte avec régularisation
    class_weights = torch.tensor([1.0, 1.5]).to(device)  # Ajuster les poids pour gérer le déséquilibre si nécessaire
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Entraînement avec Freezing des premières couches
    freeze_model_layers(model, "fc5")  # Geler les couches jusqu'à fc5

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model


#fonction qui copie les nouvelles données dans la base
def copy_eeg_files(eeg_data_list, labels_list):
    # Chemins vers les répertoires de destination
    destination_sain = "./Database Bonn/Train/Sain"
    destination_malade = "./Database Bonn/Train/Malade"


    for file_path, label in zip(eeg_data_list, labels_list):
        if os.path.exists(file_path):
            # Déterminer le répertoire de destination
            if label == 0:
                destination_dir = destination_sain
            elif label == 1:
                destination_dir = destination_malade
            else:
                print(f"Label invalide pour le fichier {file_path}: {label}")
                continue

            # Copier le fichier dans le répertoire de destination
            try:
                shutil.copy(file_path, destination_dir)
                print(f"Fichier {file_path} copié dans {destination_dir}")
            except Exception as e:
                print(f"Erreur lors de la copie du fichier {file_path}: {e}")
        else:
            print(f"Le fichier {file_path} n'existe pas.")


#fonction qui évalue la précision du modèle après l'ajout de nouvelles données
def evaluation_model(model, device, scaler):
    malade_data, malade_labels = load_data(malade_test_data_dir)
    sain_data, sain_labels = load_data(sain_test_data_dir)
    data = np.concatenate((malade_data, sain_data), axis=0)
    labels = np.concatenate((malade_labels, sain_labels), axis=0)
    # Split data using the same parameters as in training
    X_train_unused, X_test, y_train_unused, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=labels)

    # Normalize test data using the loaded scaler
    X_test_scaled = scaler.transform(X_test)

    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create test dataset and dataloader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data_batch, targets in test_loader:
            data_batch, targets = data_batch.to(device), targets.to(device)
            outputs = model(data_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Save the accuracy to a file
    with open('model_accuracy.txt', 'w') as f:
        f.write(f'{accuracy}')

    return accuracy

def load_model_and_scaler(model_path='eeg_model.pth', scaler_path='scaler.pkl'):
    model = EEGNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    try:
        with open('model_accuracy.txt', 'r') as f:
            accuracy = float(f.read())
    except FileNotFoundError:
        accuracy = evaluation_model(model, device, scaler)
        with open('model_accuracy.txt', 'w') as f:
            f.write(f'{accuracy}')
    return model, scaler, accuracy

#---------------------------------fin des fonctions utiles pour l'IA----------------------------------------------




def on_closing():
    app.quit()
    app.destroy()
app = App()
app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()
cursor.close()
conn.close()
