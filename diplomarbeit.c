
/*n13akaibe:
Diplomarbeit Trainingsteil ANN
Benedikt Kaiser
gcc -o da da.c
./da

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <syslog.h>
#include <math.h>
#include <float.h>
#define DEBUGMODE     0
#define AMNT_LAYER    3
#define AMNT_NEURON_PER_LAYER 4
#define AMNT_NEURON   ( AMNT_LAYER*AMNT_NEURON_PER_LAYER )
#define LERNRATE  0.06F
#define TRAINING_STEPS  500
#define AMNT_INPUTS   9
#define AMNT_OUTPUTS    1
#define DIFFERENTPATTERNS 2
#define LERNPARAMETER 0.000001F


  /*'SM'...SynapsenMatrix, 'r'...row, 'c'...column: */
#define DIMc_SM     ( AMNT_LAYER*AMNT_NEURON_PER_LAYER + AMNT_OUTPUTS )
#define DIMr_SM     ( AMNT_LAYER*AMNT_NEURON_PER_LAYER + AMNT_INPUTS )
double neuronen_yi_Array[AMNT_NEURON];
double neuronen_xi_Array[AMNT_NEURON];
double inputArray[AMNT_INPUTS];
double outputArray[AMNT_OUTPUTS];
double eingangswerte[DIFFERENTPATTERNS][AMNT_INPUTS] = {{0.9,0.1,0.9,0.1,0.9,0.1,0.9,0.1,0.9},{0.9,0.9,0.9,0.9,0.1,0.9,0.9,0.9,0.9}};
double neuron_soll[DIFFERENTPATTERNS][AMNT_OUTPUTS]={{0.1},{0.9}};
double testeingang[AMNT_INPUTS] = {0.9,0.9,0.9,0.1,0.1,0.1,0.9,0.9,0.9};
double trainingsInputArray[DIFFERENTPATTERNS][AMNT_INPUTS];
double trainingsOutputArray[DIFFERENTPATTERNS][AMNT_OUTPUTS];
static char TRAINING_INPUT_FILE []= "Trainingsdaten/inputDaten.txt";
static char TRAINING_OUTPUT_FILE []= "Trainingsdaten/outputDaten.txt";
static char INPUT_FILE []= "input.CSV";
static char OUTPUT_FILE []= "output.txt";
int patternnummer;
float testabweichung;
#define TRANSFER_XMIN ((double) -5.0 )
#define TRANSFER_XMAX   ((double) +5.0 )
#define TRANSFER_YMIN ((double) -1.0 )
#define TRANSFER_YMAX   ((double) +1.0 )
double transfer_XMIN=TRANSFER_XMIN,
     transfer_XMAX=TRANSFER_XMAX,
     transfer_YMIN=TRANSFER_YMIN,
     transfer_YMAX=TRANSFER_YMAX;
double threshld [ DIMc_SM];

void initialisieren();
int forwardpass();
int differenz(int,int);
int backpropagation(int,float,int);
int zeros();
int muster(int);
int lernen(int);
void onerun();
double aktivierungsfunktion(double);
double uebertragungsfunktion(int);
double zufzi(double, double);
void get_inputs(char*);
void get_trainings_daten(char*, char*);
void writeOutput(char *, int);
void TESTaktivierungsfunktion();
void TESTuebertragungsfunktion();

int trainingsmode;

float SM[DIMr_SM] [DIMc_SM];
double  neuron_ist [DIMc_SM];


void prSM(){ int r,c; printf("\nSynMat: [%d x %d]\n", DIMr_SM+1, DIMc_SM+1);
  for(r=0; r< DIMr_SM; r++){
     printf("row[%d]: ",r);
    for(c=0; c<DIMc_SM; c++)printf("%4.2f ", SM[r][c]);
    printf("\n");
} }


int zeros(){
  int r,c, zero=0;
  for(r=0;r<=DIMr_SM;r++){
  for(c=0;c<=DIMc_SM;c++){
  if(SM[r][c]==0.0)zero++;}}
printf("Die Matrix hat %d Nullen",zero);
}

double genZufallszahl(double von, double bis) {
  static int zufzi_initialisiert = 0;
  if (!zufzi_initialisiert)srand(997);
  return(von + (bis - von) * (rand() / (double)RAND_MAX));
}

void initialisieren() {
    int i;
        int  j;
        
    for (i = 0; i < DIMr_SM; i++) {
        for (j = 0; j < DIMc_SM; j++) {
            SM[j][i] = 0.0;
        }
    }
    for (i=0; i < AMNT_INPUTS; i++) {
          int j = 0;
      for (j; j < AMNT_NEURON_PER_LAYER; j++) {
        SM[i][j] = genZufallszahl(-0.8, 0.8);
      }
    }
    int m = 0;
    for (m; m < AMNT_LAYER - 1; m++){
      int von = m * AMNT_NEURON_PER_LAYER;
      int bis = (m + 1) * AMNT_NEURON_PER_LAYER;
      int n = von;
      for (n; n < bis; n++) {
              int l = von;
        for (l; l < bis; l++) {
          SM[n+AMNT_INPUTS][l+AMNT_NEURON_PER_LAYER] = genZufallszahl(-1.0, 1.0);
        }
      }
    }
    int k = DIMr_SM - AMNT_NEURON_PER_LAYER;
    for (k; k < DIMr_SM; k++) {
        int f = AMNT_NEURON;
   
     for (f; f < DIMc_SM; f++) {
        SM[k][f] = genZufallszahl(-1.0, 1.0);
      }
    }

  /*Nguyen-Widrow Zufallszahlen*/
    double new_weight=0;
    double summe_weight=0;
    double b=0.7*AMNT_NEURON/AMNT_INPUTS;

    for(k=0;k<AMNT_LAYER;k++){
      for(i=0;i<DIMr_SM;i++){
        summe_weight+= SM[i][k]*SM[i][k];
      }
      new_weight=sqrt(summe_weight);
    }

   for (i=0; i < AMNT_INPUTS; i++) {
          int j = 0;
      for(j; j < AMNT_NEURON_PER_LAYER; j++) {
        //printf("B Beträgt: %4.2f\nSM[i][j] beträgt: %4.2f\nnew_weight beträgt %4.2f\n-------------------\n",b,SM[i][j],new_weight);
        SM[i][j] = (b*SM[i][j])/new_weight;
      }
    }

    for (m=0; m < AMNT_LAYER - 1; m++){
       int von = m * AMNT_NEURON_PER_LAYER;
       int bis = (m + 1) * AMNT_NEURON_PER_LAYER;
       int n = von;
      for (n; n < bis; n++) {
              int l = von;
        for (l; l < bis; l++) {
          SM[n+AMNT_INPUTS][l+AMNT_NEURON_PER_LAYER] = (b*SM[n+AMNT_INPUTS][l+AMNT_NEURON_PER_LAYER])/new_weight;
      }
    }
  }
    for (k = DIMr_SM - AMNT_NEURON_PER_LAYER; k < DIMr_SM; k++) {
        int f = AMNT_NEURON;
   
     for (f; f < DIMc_SM; f++) {
        SM[k][f] = (b*SM[k][f])/new_weight;
      }
    }
  }

  void sm_nullsetzen() {

  int i, j;

  for (i = 0; i < DIMr_SM; i++) {
    for (j = 0; j < DIMc_SM; j++) {
      SM[i][j] = 0.0;
    }
  }
}

void onerun (){
    int s, l, y;
    int x = AMNT_INPUTS;
    double summe, yi = 0;
    
    for(l=0; l< AMNT_LAYER+1; l++){
        for (s=0; s < DIMc_SM; s++){
            if(s < AMNT_NEURON){
               summe = uebertragungsfunktion(s);
                yi = aktivierungsfunktion(summe);
                neuronen_yi_Array[s] = yi;
               // printf("yi array: %f\n", neuronen_yi_Array[s]);
            }else{
                outputArray[s-AMNT_NEURON]= uebertragungsfunktion(s);
                //printf("Es geht");
                
            }
        }
           
               for(y=0; y < AMNT_NEURON; y++){
                 neuronen_xi_Array[y] = neuronen_yi_Array[y];
               }
            
    } 
}

double uebertragungsfunktion(int spaltendindex) {

  double summe = 0;
  double eingang;
  int r = 0;
    for (r; r < DIMr_SM; r++) {
              if(r < AMNT_NEURON){
                  eingang = inputArray[r];
              }else{
                  eingang = neuronen_xi_Array[r - AMNT_NEURON];
              }
             
      summe += eingang * SM[r][spaltendindex];
    }

  return summe;
}




double aktivierungsfunktion(double inpt_summe){

    return(1/(1 + exp(-inpt_summe)));
}
void write_output(char* outputFile, int musternr) {

    FILE *fp;
    int i,j;
    if (!(fp = fopen(outputFile, "a"))) {
            perror("File konnte nicht geoeffnet werden");
    }
    else {
        fprintf(fp,"Muster %i Output:\r\n", musternr);
        for (i = 0; i < AMNT_OUTPUTS; i++) {           
            fprintf(fp, "%g,", outputArray[i]);
        }
        fprintf(fp,";\r\n");
    }
}

void get_trainingsdaten(char* trainingsInputFile, char* trainingsOutputFile){
    
    FILE *fp;
    int i,j,m,n;
    char citemp;
    
    if (!(fp = fopen(trainingsInputFile, "r"))) {
        perror("Datei konnte nicht gelesen werden");
    }
    else {
        i = 0;
        j = 0;
        n = 0;
        char *carray = (char*)malloc(sizeof(char) * 20);
        while (!feof(fp)) {            
            citemp = getc(fp);
            
            if(citemp != ';'){                
                if(citemp != ','){ 
                    carray[j] = citemp;
                    j++;                       
                }else{                    
                    trainingsInputArray[n][i] = atof(carray);
                    for(m = 0; m < j; m++){
                        carray[m] = 0;
                    }
                    j = 0;
                    i++;
                }
            }else{
                n++;
                i = 0;
            }
        }
    }
    fclose(fp);


    FILE *f;
    int u,v,s,t;
    char cotemp;
    if (!(f = fopen(trainingsOutputFile, "r"))) {
        perror("Datei konnte nicht gelesen werden");
    }
    else {
        u = 0;
        v = 0;
        t = 0;
        char *carray = (char*)malloc(sizeof(int) * 20);
        while (!feof(f)) {
            cotemp = getc(f);
            
            if(cotemp != ';'){
                if(cotemp != ','){ 
                    carray[v] = cotemp;
                    v++;                       
                }else{
                    trainingsOutputArray[t][u] = atof(carray);
                    for(s = 0; s < v; s++){
                        carray[s] = 0;
                    }
                    v = 0;
                    u++;
                }
            }else{
                t++;
                u = 0;
            }   
        }
    }
    fclose(f);
}

int muster(int trainingsmode){
int i,k,j;
get_trainingsdaten(TRAINING_INPUT_FILE,TRAINING_OUTPUT_FILE);
  for(i=0;i<DIFFERENTPATTERNS;i++){
    for(k=0;k<AMNT_INPUTS;k++) inputArray[k]=trainingsInputArray[i][k];
    lernen(i);
    patternnummer=i;
    write_output(OUTPUT_FILE,i);
    if(!trainingsmode){prSM();
    for(j=0;j<AMNT_OUTPUTS;j++)printf("%d. outputArray: %f\n",j+1, outputArray[j]);}
  }
return 0;
}

int lernen(int musternr){
int i;
  for(i=0;i<TRAINING_STEPS;i++){
    onerun();
    differenz(musternr,0);
     // for(i=0;i<AMNT_OUTPUTS;i++)printf("%d. outputArray: %f\n",i+1, outputArray[i]);
    //printf("\n%d.te Trainingsschritt mit dem %d.ten Muster",i,musternr+1);
  }
return 0;
}

  int differenz(int musternr,int test){
int neuron_nr, c,recursionlevel;
  float abweichung;
  for(c=0; c < AMNT_OUTPUTS; c++){
  
    abweichung = trainingsOutputArray[musternr][c]- outputArray[c];
    testabweichung=abweichung;
    if(test)printf("Die Differenz beträgt %4.2f\n",abweichung);
    if(DEBUGMODE)syslog(LOG_INFO, "%s%g\n", "Abweichung: ", abweichung);
    if(fabs(abweichung) > LERNPARAMETER){
       if(DEBUGMODE)syslog(LOG_INFO,"----------------------------------");
       if(DEBUGMODE)syslog(LOG_INFO, "%s%d\n", "Neuron: ", c);
      neuron_nr=AMNT_NEURON+c;
       backpropagation(neuron_nr, abweichung, recursionlevel=0);
      
    }

  }

}



int backpropagation(int neuron_nr, float abweichung, int recursionlevel){
  int c,r;
 float fehler; 
  if (recursionlevel > AMNT_LAYER) return 0;
  for(r=0; r < DIMr_SM; r++){
    if(fabs(SM[r][neuron_nr])>0.01){
        if(DEBUGMODE)syslog(LOG_INFO, "%s%d\n", "Backpropagationstep " , recursionlevel);
        if(DEBUGMODE)syslog(LOG_INFO, "%s%g\n", "Abweichung: ", abweichung);
      if (r< AMNT_NEURON)backpropagation(r, abweichung, recursionlevel + 1);
      SM[r][neuron_nr] += abweichung*LERNRATE;
       if(DEBUGMODE)syslog(LOG_INFO, "%s%g\n", "Gewicht: ", SM[r][neuron_nr]);
      
    }
  }
}

void get_inputs(char* inputFile){

  FILE *fp;
  int i,j,b;
  char ctemp;
  if (!(fp = fopen(inputFile, "r"))) {
    perror("Datei konnte nicht gelesen werden");
  }
  else {
            i = 0;
            j = 0;
            char *carray = (char*)malloc(sizeof(int) * 20);
    while (!feof(fp)) {
                    ctemp = getc(fp);
                    if(ctemp != ','){ 
                        carray[j] = ctemp;
                        j++;                       
                    }else{
                    inputArray[i] = atof(carray);
                    for(b = 0; b < j; b++){
                        carray[b] = 0;
                    }
                    j = 0;
                    i++;
                    }
    }
  }
  fclose(fp);
}




void TESTgenZufallszahl() {
    
    double von[4] = {-DBL_MAX, -10, -5, 0};
    double bis[4] = {DBL_MAX, 10, 0, 5};
    int i;
    for(i = 0; i < 4; i++){
        
        double ausg;
        printf("Eingabewert von: %g \nEingabewert bis: %g\n", von[i], bis[i]);
        ausg = genZufallszahl(von[i], bis[i]);
        printf("==> Ausgabe: %g\n\n", ausg);
    }
}
void TESTaktivierungsfunktion(){
    
    double testarray[3] = {DBL_MAX, 0, -DBL_MAX};
    int i;
    for(i = 0; i < 3; i++){
        double res;
        printf("Eingabewert: %g\n", testarray[i]);
        res = aktivierungsfunktion(testarray[i]);
        printf("==> Ausgabe: %g\n", res);
    }
}

void TESTuebertragungsfunktion(){

  double testarrayinput [DIMr_SM];
  double testarraysmr [DIMr_SM];
  double eingabewerte [5] = {sqrt(DBL_MAX/DIMr_SM),-2,0,2,-sqrt(DBL_MAX/DIMr_SM)};
  int i,j,r;

  for(i = 0; i < 5; i++){    

    printf("Eingabewert: %g\n", eingabewerte[i]);
    double sollausgabe = eingabewerte[i] * eingabewerte[i] * (DIMr_SM - 0.00001);
    printf("Soll-Ausgabe: %g\n", sollausgabe);
    double summe = 0;
    for(j = 0; j < DIMr_SM; j++){

      testarrayinput[j] = eingabewerte[i];
      testarraysmr[j] = eingabewerte[i];
    }  
    for (r= 0; r < DIMr_SM; r++) {

      summe += testarrayinput[r] * testarraysmr[r];
    }      
    printf("==> Ausgabe: %g\n", summe);
  }
}

void TESTsm_nullsetzen(){
  printf("Funktion aufrufen: sm_nullsetzen();\n");
  sm_nullsetzen();
  printf("==> Ausgabe:");
  prSM();
}

void TESTinit_sm(){
  printf("Funktion aufrufen: init_sm();\n");
  initialisieren();
  printf("==> Ausgabe:");
  prSM();
}

void TESTget_trainingsdaten(){

    printf("Funktion aufrufen: get_trainingsdaten(TRAINING_INPUT_FILE, TRAINING_OUTPUT_FILE);\n");
    get_trainingsdaten(TRAINING_INPUT_FILE, TRAINING_OUTPUT_FILE);
    
    int i,j,n;
    
    printf("==> Ausgabe:\n");
    for(i = 0; i < DIFFERENTPATTERNS; i++){
        printf("Pattern Nr. : %i", i);
        printf("\nInput Pattern (%i): \n", i);
        for(j = 0; j < AMNT_INPUTS; j++){
            
            printf("    %g\n", trainingsInputArray[i][j]);
        } 
        printf("\nOutput Pattern (%i):\n", i);
        for(n = 0; n < AMNT_OUTPUTS; n++){
           
            printf("    %g\n", trainingsOutputArray[i][n]);
        }
    }
}

void TESTwrite_output(){

  int i,j;
  for (i = 0; i < AMNT_OUTPUTS; i++){

    outputArray[i] = i;
  }
  printf("Outputarray: ");
  for (j = 0; j < AMNT_OUTPUTS; j++){

    printf("%g,", outputArray[j]);
  }
  printf("\nFunktion aufrufen: write_output(OUTPUT_FILE, 1);\n");
  write_output(OUTPUT_FILE, 1);
  printf("Vergleichen Sie nun die Ausgabe in der Datei %s mit den Werten im outputArray.\n", OUTPUT_FILE);
}

void TESTgetInput() {
  printf("Funktion aufrufen: get_inputs(INPUT_FILE);\n");
  get_inputs(INPUT_FILE);

  printf ("==> Ausgabe:");

        int i;
  for (i = 0; i < AMNT_INPUTS; i++) {
            printf("%g,", inputArray[i]);
  }
  printf("\n");
}
   
void TESTdiff(){
  int i;
  initialisieren();
  onerun();
  get_trainingsdaten(TRAINING_INPUT_FILE, TRAINING_OUTPUT_FILE);
 differenz(patternnummer,1);
for(i=0;i<AMNT_OUTPUTS;i++)printf("\nAusgang: %4.2f\n",outputArray[i]);
printf("\n Stimmt die Differenz?");
printf("\n %4.2f + %4.2f = %4.2f\n",testabweichung,outputArray[0],trainingsOutputArray[0][0]);
}

void TESTbackprop(){
  int j;
  get_trainingsdaten(TRAINING_INPUT_FILE, TRAINING_OUTPUT_FILE);
  printf("Vor dem Lernen: \n");
  printf("  Soll: %4.2f\n",trainingsOutputArray[1][0]);
  printf("  Ist: %4.2f\n", outputArray[0]);
  initialisieren();
  muster(1);
  printf("Nach dem Lernen: \n");
  printf("  Soll: %4.2f\n",trainingsOutputArray[1][0]);
  printf("  Ist: %4.2f\n", outputArray[0]);}

int helptext(char **argv){
  printf("usage:\n");
  printf("%s -i   Netzwerkinitialisierung\n", argv[0]);
  printf("%s -p   Netzwerkausgabe\n", argv[0]);
  printf("%s -a   Errechne Ausgang\n", argv[0]);
  printf("%s -t   Trainiere Netzwerk\n", argv[0]);
  printf("%s -k   Test Aktivierungsfunktion\n", argv[0]);
  printf("%s -z   Test Zufallszahl\n", argv[0]);
  printf("%s -u   Test Uebertragungsfunktion\n", argv[0]);
  printf("%s -n   Test SM Nullsetzen\n", argv[0]);
  printf("%s -s   Test Initialisierung\n",argv[0]);
  printf("%s -g   Test Trainingsdaten einlesen\n",argv[0]);
  printf("%s -w   Test Writeoutput\n",argv[0]);
  printf("%s -c   Test GetInput\n",argv[0]);
  printf("%s -d   Test Differenz\n",argv[0]);
  printf("%s -r   Test Backpropagation\n",argv[0]);
}

int main_parameter_auswerten(int argc, char **argv){
 int  i,r,opt_rv;

 if(argc<2) exit(0); 

  switch(argv[1][1]){
  case 'i' : initialisieren();prSM();break;
  case 'p' : prSM();break;
  case 'a' : initialisieren();onerun();for(i=0;i<AMNT_OUTPUTS;i++)printf("Ausgang: %4.2f\n",outputArray[i]);break;
  case 't' : initialisieren();muster(0); break;
  case 'k' : TESTaktivierungsfunktion();break;
  case 'z' : TESTgenZufallszahl(); break;
  case 'u' : TESTuebertragungsfunktion();break;
  case 'n' : TESTsm_nullsetzen(); break;
  case 's' : TESTinit_sm();break;
  case 'g' : TESTget_trainingsdaten();break;
  case 'w' : TESTwrite_output();break;
  case 'c' : TESTgetInput();break;
  case 'd' : TESTdiff();break;
  case 'r' : TESTbackprop();break;
  default: ;
  }
 
}

int main(int argc, char**argv){
  int i,k;
  if(argc<2){ helptext(argv); exit(0);}
    main_parameter_auswerten(argc,argv);
  /*initialisieren();
  prSM();
  muster();
 for(k=0;k<AMNT_INPUTS;k++) inputArray[k]=testeingang[k];
      onerun();
     for(k=0;k<AMNT_OUTPUTS;k++)printf("%d. outputArray mit Testeingang: %f\n",k+1, outputArray[k]);
*/
 return 0;
}
/*EOF*/


