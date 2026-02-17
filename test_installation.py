#!/usr/bin/env python3
"""
Script de vérification de l'installation du projet.

Ce script vérifie que toutes les dépendances sont correctement installées
et que l'environnement est prêt pour l'exécution des notebooks.

Usage:
    python test_installation.py
"""

import sys
import importlib
from typing import List, Tuple

def check_python_version() -> bool:
    """Vérifie la version de Python."""
    required_version = (3, 10)
    current_version = sys.version_info[:2]
    
    print(f"🐍 Python version: {current_version[0]}.{current_version[1]}")
    
    if current_version >= required_version:
        print("   ✅ Version compatible")
        return True
    else:
        print(f"   ❌ Python {required_version[0]}.{required_version[1]}+ requis")
        return False


def check_packages() -> Tuple[List[str], List[str]]:
    """Vérifie l'installation des packages requis."""
    
    required_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'plotly',
        'sklearn',
        'xgboost',
        'scipy',
        'jupyter',
        'tqdm'
    ]
    
    installed = []
    missing = []
    
    print("\n📦 Vérification des packages:")
    print("-" * 50)
    
    for package in required_packages:
        try:
            # Cas spécial pour scikit-learn
            if package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            
            # Obtenir la version si possible
            try:
                if package == 'sklearn':
                    module = importlib.import_module('sklearn')
                else:
                    module = importlib.import_module(package)
                version = getattr(module, '__version__', 'N/A')
                print(f"   ✅ {package:<15} version {version}")
                installed.append(package)
            except:
                print(f"   ✅ {package:<15} (version inconnue)")
                installed.append(package)
                
        except ImportError:
            print(f"   ❌ {package:<15} NON INSTALLÉ")
            missing.append(package)
    
    return installed, missing


def check_directory_structure() -> bool:
    """Vérifie la structure des dossiers."""
    import os
    
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'models'
    ]
    
    print("\n📁 Vérification de la structure:")
    print("-" * 50)
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ⚠️  {directory} (sera créé automatiquement)")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"      ✅ Créé avec succès")
            except Exception as e:
                print(f"      ❌ Erreur: {e}")
                all_exist = False
    
    return all_exist


def check_jupyter() -> bool:
    """Vérifie que Jupyter est accessible."""
    import subprocess
    
    print("\n📓 Vérification de Jupyter:")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            ['jupyter', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("   ✅ Jupyter est installé et accessible")
            print(f"   Version: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Problème avec Jupyter")
            return False
    except FileNotFoundError:
        print("   ❌ Jupyter non trouvé dans le PATH")
        return False
    except subprocess.TimeoutExpired:
        print("   ⚠️  Timeout lors de la vérification")
        return False
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def test_basic_operations() -> bool:
    """Teste quelques opérations de base."""
    print("\n🧪 Test des opérations de base:")
    print("-" * 50)
    
    try:
        # Test pandas
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print("   ✅ Pandas DataFrame: OK")
        
        # Test numpy
        import numpy as np
        arr = np.array([1, 2, 3])
        mean = arr.mean()
        print("   ✅ NumPy operations: OK")
        
        # Test sklearn
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        print("   ✅ Scikit-learn import: OK")
        
        # Test plotly
        import plotly.graph_objects as go
        fig = go.Figure()
        print("   ✅ Plotly import: OK")
        
        # Test xgboost
        import xgboost as xgb
        print("   ✅ XGBoost import: OK")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur lors des tests: {e}")
        return False


def print_installation_instructions(missing_packages: List[str]) -> None:
    """Affiche les instructions d'installation pour les packages manquants."""
    if not missing_packages:
        return
    
    print("\n" + "=" * 70)
    print("📥 INSTRUCTIONS D'INSTALLATION DES PACKAGES MANQUANTS")
    print("=" * 70)
    
    print("\nOption 1 - Installation via requirements.txt (RECOMMANDÉ):")
    print("-" * 70)
    print("pip install -r requirements.txt")
    
    print("\nOption 2 - Installation manuelle:")
    print("-" * 70)
    for package in missing_packages:
        if package == 'sklearn':
            print(f"pip install scikit-learn")
        else:
            print(f"pip install {package}")
    
    print("\nOption 3 - Installation avec conda:")
    print("-" * 70)
    for package in missing_packages:
        if package == 'sklearn':
            print(f"conda install scikit-learn")
        else:
            print(f"conda install {package}")


def generate_summary(checks: dict) -> None:
    """Génère un résumé des vérifications."""
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE LA VÉRIFICATION")
    print("=" * 70)
    
    total_checks = len(checks)
    passed_checks = sum(1 for v in checks.values() if v)
    
    print(f"\nRésultat: {passed_checks}/{total_checks} vérifications réussies")
    print()
    
    for check_name, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"   {icon} {check_name}")
    
    print("\n" + "=" * 70)
    
    if passed_checks == total_checks:
        print("🎉 TOUT EST PRÊT!")
        print("\nVous pouvez maintenant:")
        print("   1. Lancer Jupyter: jupyter notebook")
        print("   2. Ouvrir: notebooks/01_data_generation.ipynb")
        print("   3. Exécuter les cellules: Cell > Run All")
    else:
        print("⚠️  CONFIGURATION INCOMPLÈTE")
        print("\nVeuillez installer les packages manquants avant de continuer.")
        print("Consultez le fichier README.md pour plus d'informations.")
    
    print("=" * 70)


def main():
    """Fonction principale."""
    print("=" * 70)
    print("🔍 VÉRIFICATION DE L'INSTALLATION")
    print("   Projet: Analyse du Marché Automobile")
    print("=" * 70)
    
    # Dictionnaire pour stocker les résultats
    checks = {}
    
    # 1. Vérifier Python
    checks['Python 3.10+'] = check_python_version()
    
    # 2. Vérifier les packages
    installed, missing = check_packages()
    checks['Packages requis'] = len(missing) == 0
    
    # 3. Vérifier la structure
    checks['Structure des dossiers'] = check_directory_structure()
    
    # 4. Vérifier Jupyter
    checks['Jupyter Notebook'] = check_jupyter()
    
    # 5. Tests de base
    checks['Opérations de base'] = test_basic_operations()
    
    # Afficher les instructions si nécessaire
    if missing:
        print_installation_instructions(missing)
    
    # Résumé final
    generate_summary(checks)
    
    # Code de sortie
    exit_code = 0 if all(checks.values()) else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Vérification interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
