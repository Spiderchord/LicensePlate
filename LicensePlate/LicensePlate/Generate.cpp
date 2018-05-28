// Generate.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

//globalne zmienne
const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 30;
const int RESIZED_IMAGE_HEIGHT = 40;
/*

int main() {
	//obrazek na wejsciu
    cv::Mat imgTrainingNumbers;
    //obrazek w skali szarosci
	cv::Mat imgGrayscale;
	// obrazek z blurrem
    cv::Mat imgBlurred; 
	//obrazek sthresholdowany
    cv::Mat imgThresh;
	//musi byc kopia
    cv::Mat imgThreshCopy; 

	//wektor konturow
    std::vector<std::vector<cv::Point> > ptContours;  
	//hierarchia konturow
    std::vector<cv::Vec4i> v4iHierarchy;                  

	//classification numbers do uczenia
    cv::Mat matClassificationInts;      
										
    //obrazki do uczenia w formacie 1 wiersz
	cv::Mat matTrainingImagesAsFlattenedFloats;

    //mozliwe znaki do wykrywania
    std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z' };

	//wczytywanie obrazka z znakami z tablic do uczenia
    imgTrainingNumbers = cv::imread("znaki4.png");         

	//jesli nie mozna otworzyc obrazka to wywal error
    if (imgTrainingNumbers.empty()) {                               
        std::cout << "Nie mozna wczytac obrazka z pliku";         
        return(0);                                                  
    }

	//konwersja do skali szarosci (potrzebne do thresholdu)
    cv::cvtColor(imgTrainingNumbers, imgGrayscale, CV_BGR2GRAY);

	//blurr Gaussa
    cv::GaussianBlur(
		imgGrayscale,              //obrazek wejsciowy
        imgBlurred,                             //obrazek wyjsciowy
        cv::Size(5, 5),                         // rozmiar kernela gaussowskiego
        0);                                     // to samo co kernel

    //najlepszy threshold, szczegolnie przy roznych warunkach oswietlenia, zmienia obraz na czarno bialy                                           
    cv::adaptiveThreshold(
		imgBlurred,           //wejsciowy obrazek zblurrowany wyzej
        imgThresh,                              //obrazek wyjsciowy sthresholdowany
        255,                                    // zmien piksel na bisaly gdy jego wartosc przekroczy wartosc thresholdu
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // chyba lepszy niz MEAN
        cv::THRESH_BINARY_INV,                  // przod bialy a tlo czarne
        11,                                     // rozmiar sasiednich obszarow
        2);                                     // stala

	//pokaz obrazek po thresholdzie
    cv::imshow("obrazek po thresholdzie", imgThresh);     

	//kopia obrazka, bo metoda findcontours modyfikuje obrazek 
    imgThreshCopy = imgThresh.clone();        

    cv::findContours(
		imgThreshCopy,             //kopia obrazka, bo modyfikuje go
        ptContours,                             //wyjsciowe kontury
        v4iHierarchy,                           //wyjsciowa hierarchia konturow
        cv::RETR_EXTERNAL,                      //wyciaganie tylko konturow zewnetrznych
        cv::CHAIN_APPROX_SIMPLE);               //do zaoszczedzenia pamieci ram

	//jedziemy po konturach
    for (int i = 0; i < ptContours.size(); i++) { 
		//jesli kontur jest wiekszy niz 100
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {   
			//pobieramy bounding rectangle
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);

			//rysowanie kwadratu dookola kazdego konturu
            cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);

			//pobranie znaku wycietego i uzycie thresholdu na nim
            cv::Mat matROI = imgThresh(boundingRect);

            cv::Mat matROIResized;

			//zmiana rozmiaru wycietego znaku do 30x40
            cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

			//wyswietlanie znaku, zeby bylo wiadomo co sie dzieje
            cv::imshow("matROI", matROI);   
			//wyswietlenie znaku po resize
            cv::imshow("matROIResized", matROIResized); 
			//wyswietlenie obrazka ze znakami z kwadratem dookola nich (w celu uczenia)
            cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       

			//oczekiwanie na wcisniecie odpowiedniego znaku na klawiaturze (jesli jest podswietlony to wciskamy jego odpowiednik)
            int intChar = cv::waitKey(0);           

			//wcisnij escape aby wyjsc
            if (intChar == 27) {        
                return(0);             
            }
			//gdy wcisniety znak jest na liscie znakow ktore chcemy rozpoznawac
            else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {     

				//dopisz znak do listy
                matClassificationInts.push_back(intChar);

                cv::Mat matImageFloat;        
				//zmiana Mat na 32bit float, 1 kanal
                matROIResized.convertTo(matImageFloat, CV_32FC1);       // convert Mat to float

				// flatten, argumenty = liczba kanalow i liczba wierszy, wszystko w jednym wierszu w celu redukcji czasu dzialalnia programu
                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1); 

				//dodaje elementy na dol macierzy
                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       
                                                                                            
            } 
        }  
    } 


    //zapisujemy classification numbers do pliku

	//otwarcie pliku do zapisu
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);

	//jesli nie mozna otworzyc to za wywal prograsm
    if (fsClassifications.isOpened() == false) {                                                        
        std::cout << "Nie mozna otworzyc pliku";        
        return(0);                                                                                      
    }

	//zapisywanie classification numbers do pliku
    fsClassifications << "classifications" << matClassificationInts; 

	//zamknij plik
    fsClassifications.release();                                            

	//zapisywanie znakow do uczenia do pliku

    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         

    if (fsTrainingImages.isOpened() == false) {                                                 
        std::cout << "Nie mozna otworzyc pliku";         
        return(0);                                                                              
    }

	//zapisywanie znakow do uczenia do pliku
    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;         
    fsTrainingImages.release();                                                 

    return(0);
}

*/


