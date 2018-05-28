// DetectPlates.cpp

#include "DetectPlates.h"

//wykrywanie tablic na obrazku
std::vector<PossiblePlate> detectPlatesInScene(cv::Mat &imgOriginalScene) {
	//zwrocony vectorOfPossiblePlates
    std::vector<PossiblePlate> vectorOfPossiblePlates;			

    cv::Mat imgGrayscaleScene;
    cv::Mat imgThreshScene;
    cv::Mat imgContours(imgOriginalScene.size(), CV_8UC3, SCALAR_BLACK);

    cv::RNG rng;

    cv::destroyAllWindows();

	//preprocessing, grayscale i threshold
    preprocess(imgOriginalScene, imgGrayscaleScene, imgThreshScene);

    // znajdowanie wszystkich mozliwych znakow na obrazie, funkcja znajduje wszystkie kontury i dodaje do vectora
	// tylko te, ktore moga byc znakami (bez porownywania znakow)
    std::vector<PossibleChar> vectorOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene);

	//podajemy vector wszystkich mozliwych znakow, funkcja znajduje grupy pasujacych znakow, w nastepnych krokach
	// kazda grupa mozliwych znakow bedzie poddawana rozpoznawaniu jako tablica
    std::vector<std::vector<PossibleChar> > vectorOfVectorsOfMatchingCharsInScene = findVectorOfVectorsOfMatchingChars(vectorOfPossibleCharsInScene);

    for (auto &vectorOfMatchingChars : vectorOfVectorsOfMatchingCharsInScene) {                     //dla kazdej grupy pasujacych znakow
        PossiblePlate possiblePlate = extractPlate(imgOriginalScene, vectorOfMatchingChars);        //proba wyekstraktowania tablicy

        if (possiblePlate.imgPlate.empty() == false) {                                              //jesli tablica zostala znaleziona
            vectorOfPossiblePlates.push_back(possiblePlate);                                        //dodaj do vectorOfPossiblePlates
        }
    }

    std::cout << std::endl << vectorOfPossiblePlates.size() << " mozliwych tablic" << std::endl;       //wyswietl ilosc mozliwych tablic

	//zwracamy vector mozliwych tablic
    return vectorOfPossiblePlates;
}

std::vector<PossibleChar> findPossibleCharsInScene(cv::Mat &imgThresh) {
    std::vector<PossibleChar> vectorOfPossibleChars;            //zwracamy vectorOfPossibleChars

    cv::Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);
    int intCountOfPossibleChars = 0;

    cv::Mat imgThreshCopy = imgThresh.clone();

    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(imgThreshCopy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);        //szukanie wszystkich konturow

    for (unsigned int i = 0; i < contours.size(); i++) {                //dla kazdego konturu

        PossibleChar possibleChar(contours[i]);

        if (checkIfPossibleChar(possibleChar)) {                // jesli kontur jest mozliwym znakiem
            intCountOfPossibleChars++;                          // zwieksz licznik ilosci mozliwych znakow
            vectorOfPossibleChars.push_back(possibleChar);      // dodaj do vectorOfPossibleChars
        }
    }

	//zwracamy vector mozliwych znakow
    return(vectorOfPossibleChars);
}

PossiblePlate extractPlate(cv::Mat &imgOriginal, std::vector<PossibleChar> &vectorOfMatchingChars) {
    PossiblePlate possiblePlate;            //zwracamy mozliwa tablice

                                            //sortowanie znakow od lewej do prawej bazujac na pozycji x
    std::sort(vectorOfMatchingChars.begin(), vectorOfMatchingChars.end(), PossibleChar::sortCharsLeftToRight);

    // obliczanie centralnego punktu tablicy
    double dblPlateCenterX = (double)(vectorOfMatchingChars[0].intCenterX + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterX) / 2.0;
    double dblPlateCenterY = (double)(vectorOfMatchingChars[0].intCenterY + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY) / 2.0;
    cv::Point2d p2dPlateCenter(dblPlateCenterX, dblPlateCenterY);

    // obliczanje dlugosci i wysokosci tablicy
    int intPlateWidth = (int)((vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.x + vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].boundingRect.width - vectorOfMatchingChars[0].boundingRect.x) * PLATE_WIDTH_PADDING_FACTOR);

    double intTotalOfCharHeights = 0;

    for (auto &matchingChar : vectorOfMatchingChars) {
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.boundingRect.height;
    }

    double dblAverageCharHeight = (double)intTotalOfCharHeights / vectorOfMatchingChars.size();

    int intPlateHeight = (int)(dblAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR);

    // obliczanie k¹ta tablicy
    double dblOpposite = vectorOfMatchingChars[vectorOfMatchingChars.size() - 1].intCenterY - vectorOfMatchingChars[0].intCenterY;
    double dblHypotenuse = distanceBetweenChars(vectorOfMatchingChars[0], vectorOfMatchingChars[vectorOfMatchingChars.size() - 1]);
    double dblCorrectionAngleInRad = asin(dblOpposite / dblHypotenuse);
    double dblCorrectionAngleInDeg = dblCorrectionAngleInRad * (180.0 / CV_PI);

    
    possiblePlate.rrLocationOfPlateInScene = cv::RotatedRect(p2dPlateCenter, cv::Size2f((float)intPlateWidth, (float)intPlateHeight), (float)dblCorrectionAngleInDeg);

    cv::Mat rotationMatrix;             //rotacja
    cv::Mat imgRotated;
    cv::Mat imgCropped;

    rotationMatrix = cv::getRotationMatrix2D(p2dPlateCenter, dblCorrectionAngleInDeg, 1.0);         //macierz obrotowa

    cv::warpAffine(imgOriginal, imgRotated, rotationMatrix, imgOriginal.size());            //obracanie obrazka

                                                                                            //wycinanie tablicy
    cv::getRectSubPix(imgRotated, possiblePlate.rrLocationOfPlateInScene.size, possiblePlate.rrLocationOfPlateInScene.center, imgCropped);

    possiblePlate.imgPlate = imgCropped;            //kopiowanie wycietej tablicy do possiblePlate

	//zwracamy mozliwa tablice
    return(possiblePlate);
}

