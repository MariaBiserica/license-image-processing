import cv2
import pytest
from brisque import BRISQUE

# Funcția care testează imaginea
def test_local_image():
    # Definește calea către imagine
    image_path = "..\\..\\..\\ilniqe_release_online\\IL-NIQE-master\\pepper_exa\\pepper_4.png"

    # Încarcă imaginea folosind OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Verifică dacă imaginea a fost încărcată corect
    assert img is not None, "Eroare la încărcarea imaginii. Verifică calea."

    # Creează un obiect BRISQUE
    obj = BRISQUE(url=False)

    # Calculează scorul BRISQUE
    score = obj.score(img)

    # Poți adăuga aici aserțiuni bazate pe valoarea așteptată a scorului
    # De exemplu:
    # assert score < 50, "Scorul BRISQUE este prea mare"

    print(f"\nScorul BRISQUE al imaginii este: {score}")

# Dacă vrei să păstrezi și funcția main
def main():
    test_local_image()

if __name__ == "__main__":
    main()


# import brisque
# import pytest

# def test_validate_url_score():
# from brisque import BRISQUE
# URL = "https://www.mathworks.com/help/examples/images/win64/CalculateBRISQUEScoreUsingCustomFeatureModelExample_01.png"
# obj = BRISQUE(url=True)
# assert round(obj.score(URL),3) == round(71.73427549219397, 3)
