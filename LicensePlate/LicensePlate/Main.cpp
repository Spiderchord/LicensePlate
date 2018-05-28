//Main.cpp

#include "Main.h"
#include "DetectChars.h"
#include "DetectPlates.h"

int main(void) {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();           //proba uczenia KNN

    if (blnKNNTrainingSuccessful == false) {                            // wyjdz z programu jesli sie nie udalo
                                                                       
        std::cout << std::endl << std::endl << "Uczenie nie powiodlo sie!" << std::endl << std::endl;
        return(0);                                                     
    }

    cv::Mat imgOriginalScene;           //wejsciowy obraz

    imgOriginalScene = cv::imread("obrazek6.png");         //otwieranie obrazu

    if (imgOriginalScene.empty()) {                             //jesli nie mozna otworzyc obrazu to wyjdz z programu
        std::cout << "Nie mozna odczytac obrazu z pliku!\n\n";     
        _getch();                                               
        return(0);                                              
    }

    std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(imgOriginalScene);          //wykrywanie tablic

    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);                               //wykrywanie znakow w tablicy

    cv::imshow("Oryginalny obraz", imgOriginalScene);           //pokaz obraz

    if (vectorOfPossiblePlates.empty()) {                                               //jesli nie znaleziono tablic
        std::cout << std::endl << "Nie znaleziono zadnych tablic!" << std::endl;       
    }
    else {                                                                            
        //tu mamy minimum jedna tablice
                                                                                  
		//sortowanie wektora mozliwych tablic malejaco - od najwiekszej ilosci znakow do najmniejszej
        std::sort(vectorOfPossiblePlates.begin(), vectorOfPossiblePlates.end(), PossiblePlate::sortDescendingByNumberOfChars);

		//zakladamy ze tablica z najwieksza iloscia rozpoznanych znakow (pierwsza od gory) to nasza tablica
        PossiblePlate licPlate = vectorOfPossiblePlates.front();

		//pokaz tablice
        cv::imshow("Wycieta tablica", licPlate.imgPlate);            
        cv::imshow("Thresholded", licPlate.imgThresh);

        if (licPlate.strChars.length() == 0) {                                                      //jesli nie wykryto zadnych znakow to wyjdz z programu
            std::cout << std::endl << "Nie wykryto zadnych znakow!" << std::endl << std::endl;      
            return(0);                                                                             
        }

        drawRedRectangleAroundPlate(imgOriginalScene, licPlate);                //prostokat dookola tablicy

        std::cout << std::endl << "Tablica odczytana z obrazka to " << licPlate.strChars << std::endl;     //pokaz odczytana tablice w konsoli

        writeLicensePlateCharsOnImage(imgOriginalScene, licPlate);              //wypisanie znakow z tablicy na obrazku

        cv::imshow("Oryginalny obraz", imgOriginalScene);                       

        cv::imwrite("Oryginalny obraz.png", imgOriginalScene);                  //zapisz oryginalny obraz do pliku
    }

    cv::waitKey(0);                 

    return(0);
}


void drawRedRectangleAroundPlate(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point2f p2fRectPoints[4];

    licPlate.rrLocationOfPlateInScene.points(p2fRectPoints);            // pobieramy 4 wierzcholki

    for (int i = 0; i < 4; i++) {                                       // rysujemy 4 czerwone linie
        cv::line(imgOriginalScene, p2fRectPoints[i], p2fRectPoints[(i + 1) % 4], SCALAR_RED, 2);
    }
}

void writeLicensePlateCharsOnImage(cv::Mat &imgOriginalScene, PossiblePlate &licPlate) {
    cv::Point ptCenterOfTextArea;                   // srodek obszaru gdzie wpisany bedzie text
    cv::Point ptLowerLeftTextOrigin;                // lewy dolny rog obszaru gdzie wpisany bedzie text

    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;                              
    double dblFontScale = (double)licPlate.imgPlate.rows / 30.0;            
    int intFontThickness = (int)std::round(dblFontScale * 1.5);             
    int intBaseline = 0;

    cv::Size textSize = cv::getTextSize(licPlate.strChars, intFontFace, dblFontScale, intFontThickness, &intBaseline);    

    ptCenterOfTextArea.x = (int)licPlate.rrLocationOfPlateInScene.center.x;         // the horizontal location of the text area is the same as the plate

    if (licPlate.rrLocationOfPlateInScene.center.y < (imgOriginalScene.rows * 0.75)) {      // jesli tablica jest w gornym obszarze obrazu
                                                                                            // write the chars in below the plate
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) + (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }
    else {                                                                                // jesli tablica jest w nizszym obszarze obrazu
                                                                                          // write the chars in above the plate
        ptCenterOfTextArea.y = (int)std::round(licPlate.rrLocationOfPlateInScene.center.y) - (int)std::round((double)licPlate.imgPlate.rows * 1.6);
    }

    ptLowerLeftTextOrigin.x = (int)(ptCenterOfTextArea.x - (textSize.width / 2));           
    ptLowerLeftTextOrigin.y = (int)(ptCenterOfTextArea.y + (textSize.height / 2));          

    //wypisz text na ekranie
    cv::putText(imgOriginalScene, licPlate.strChars, ptLowerLeftTextOrigin, intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
	
}


