var fso = new ActiveXObject("Scripting.FileSystemObject"); 
var main_folder = fso.GetParentFolderName(WScript.ScriptFullName);
/* 
in this main_folder had to be subfolders that match to ones
listed in array 'classes' 
*/

var classes=['-15','16-19'];

// var classes=['-15','20-29'];
// var classes=['-15','30-39'];
// var classes=['-15','40-49'];
// var classes=['-15','50+'];
// var classes=['16-19','20-29'];
// var classes=['16-19','30-39'];
// var classes=['16-19','40-49'];
// var classes=['16-19','50+'];
// var classes=['20-29','30-39'];
// var classes=['20-29','40-49'];
// var classes=['20-29','50+'];
// var classes=['30-39','40-49'];
// var classes=['30-39','50+'];
// var classes=['40-49','50+'];
// var classes=['-15','16-19','20-29','30-39','40-49','50+'];

/* 
Following array of features depend on the nature of data and 
on the results of feature selection
*/

features=["0","8","13","16","17","20","32","37","39","40","46","49","65","66","68","69","71","72","73","74","75","76","77","78","79","80","82","83","84","85","86","186","188","190","191","192","219","229","73_32","32_79","79_83","83_75","75_65","65_32","32_86","71_65","32_77","77_73","73_68","68_65","65_71","71_73","32_76","76_73","73_83","83_65","65_68","65_84","73_84","84_83","72_84","84_69","69_76","76_84","84_32","32_83","65_72","86_65","72_69","69_84","84_85","85_78","78_68","68_32","79_78","78_32","32_80","80_73","73_75","69_65","219_80","80_69","84_65","65_74","74_65","78_69","69_68","32_65","65_73","65_86","32_8","73_8","69_82","75_69","69_77","77_32","86_219","219_73","75_83","83_32","79_76","76_76","76_65","32_75","75_85","85_73","32_73","78_73","73_77","77_69","69_83","83_69","8_8","32_84","85_32","80_65","65_78","78_78","75_79","79_79","82_69","69_78","65_66","66_32","83_73","73_78","69_73","69_69","76_68","68_73","76_69","69_32","32_82","32_72","86_73","85_83","65_8","65_65","85_85","85_82","65_76","79_82","84_73","76_85","68_85","32_32","8_32","69_75","73_73","82_86","82_65","32_219","76_32","85_84","65_83","37_37","73_71","71_69","65_82","190_32","73_76","85_76","39_39","69_71","71_85","32_78","78_65","65_77","77_65","77_85","68_69","32_69","82_79","83_84","82_73","74_85","32_74","78_71","69_8","72_65","83_85","85_68","188_32","69_86","65_75","73_65","72_73","78_85","32_16","229_229","0_0","71_65_32","65_32_77","32_77_73","73_68_65","76_73_83","32_86_65","32_79_78","79_78_32","65_68_32","65_74_65","69_68_32","75_83_32","32_79_76","65_32_75","32_75_85","83_69_68","68_65_32","8_8_8","65_66_32","32_84_69","73_78_69","79_76_69","76_69_32","77_73_83","73_83_32","73_68_32","84_85_83","84_69_32","32_75_79","32_8_8","83_69_76","69_32_75","32_75_65","76_65_83","80_73_76","65_77_65","65_83_84","83_84_69","85_83_84","83_69_32","73_83_69","32_74_65","74_65_32","69_77_65","32_83_65","77_65_32","77_73_78","78_69_32","73_83_84","69_83_84","83_84_32","85_83_69","39_39_39","83_84_65","65_77_73","77_65_65","229_229_229","0_0_0","32_79_78_32","8_8_8_8","32_8_8_8","32_74_65_32","77_73_78_69","229_229_229_229","0_0_0_0"];

// features=["0","8","13","16","17","20","32","37"];
/* 
next variables set limit to seek hold (hpiir) and 
seek (spiir) times. If and how to use them depends on our intention.
For example, when we want to know about regular typing flow 
of some character combinations, then such limits are necessary,
because long interruptions in typing can dramatically change the
results. 
However, when we are interested on such patterns as 
pauses and interruptions, then we need to change those limits 
to some big values.
*/

hpiir=200;
spiir=1000;

fl=features.length;

// starting to write file for machine learning

ds=fso.CreateTextFile(main_folder+'\\data_'+classes.join('_')+'_'+fl+'.arff',true);

// next lines are for generating Weka specific headers
// after these headers the file is like regular .csv

ds.Write('@RELATION '+classes.join('-')+'\r\n\r\n');

for(i=0;i<fl;i++){
	if(features[i].indexOf('_')==-1){
		ds.WriteLine('@ATTRIBUTE hold_'+features[i]+'  NUMERIC');
		ds.WriteLine('@ATTRIBUTE seek_'+features[i]+'  NUMERIC');
	}else{
		ds.WriteLine('@ATTRIBUTE '+features[i]+'  NUMERIC');
	}
}
/*
ds.WriteLine('@ATTRIBUTE Time NUMERIC');
ds.WriteLine('@ATTRIBUTE Space2Time NUMERIC');
ds.WriteLine('@ATTRIBUTE Space3Time NUMERIC');
ds.WriteLine('@ATTRIBUTE Spaceb2Time NUMERIC');
ds.WriteLine('@ATTRIBUTE Spaceb3Time NUMERIC');
ds.WriteLine('@ATTRIBUTE Spacee2Time NUMERIC');
ds.WriteLine('@ATTRIBUTE Spacee3Time NUMERIC');
ds.WriteLine('@ATTRIBUTE Spacec3Time NUMERIC');
*/

ds.WriteLine('@ATTRIBUTE class {'+classes.join(', ')+'}\r\n\r\n@DATA');


for(i=0; i<classes.length; i++){
  var class_folder = main_folder+'\\'+classes[i];
  var CurDir = fso.GetFolder(class_folder);
  var Files = CurDir.Files;


  for(var fileitem = new Enumerator(Files); !fileitem.atEnd(); fileitem.moveNext()){

   if(fileitem.item().Name.indexOf('.csv')!=-1){

    file_name=fileitem.item().Name;
    WScript.Echo("Processing log: "+file_name);

     fail = fso.OpenTextFile(class_folder+"\\"+file_name,1);
     log_id=file_name.split('.txt').join('');

     sisu = fail.ReadAll();
     fail.Close();
     
     rida=''; 
     m=csv2arr(sisu);
     ml=m.length-1;
     klahvall=0;
     klvahe=0;
     hold=[];
     seek=[];
     fv={};
     fvc={};
     for(f=0;f<fl;f++){
      rida='';
      for(hc=0,sc=0,j=0;j<ml;j++){

// hold       
         if(parseInt(m[j][1])==m[j][1]&&m[j][1]<hpiir){
          
          klahvall+=m[j][1];
          hc++;
          hold.push(m[j][1]);
          if(features[f]==m[j][0]+""){
	  	if (fv['hold_'+features[f]]) {
           		fv['hold_'+features[f]]+=m[j][1];
          	} else {
           		fv['hold_'+features[f]]=m[j][1];
          	}
		if(fvc['hold_'+features[f]]){
			fvc['hold_'+features[f]]++;
		}else{
			fvc['hold_'+features[f]]=1;
		}
          }

         }

// seek
         if(parseInt(m[j][2])==m[j][2]&&m[j][2]<spiir){
             if(features[f]*1==m[j][0]){
          
	       if (fv['seek_'+features[f]]) {
                 	fv['seek_'+features[f]]+=m[j][2];
                } else {
                 	fv['seek_'+features[f]]=m[j][2];
                }
		if(fvc['seek_'+features[f]]){
		  	fvc['seek_'+features[f]]++;
		}else{
		  	fvc['seek_'+features[f]]=1;
		}
	     }
         }

// n-graphs

          if(j<ml-1){

            if(features[f]==m[j][0]+'_'+m[j+1][0]&&(m[j][1]+m[j+1][1]+m[j+1][2])<spiir*2){
	        if (fv[features[f]]) {
                  	fv[features[f]]+=m[j][1]+m[j+1][1]+m[j+1][2];
                } else {
                  	fv[features[f]]=m[j][1]+m[j+1][1]+m[j+1][2];
                }
		if(fvc[features[f]]){
		  	fvc[features[f]]++;
		}else{
		  	fvc[features[f]]=1;
		}

            }
              
	  }

	if(j<ml-2){

          if(features[f]==m[j][0]+'_'+m[j+1][0]+'_'+m[j+2][0]&&(m[j][1]+m[j+1][1]+m[j+1][2]+m[j+2][1]+m[j+2][2])<spiir*3){
	        if (fv[features[f]]) {
                  	fv[features[f]]+=m[j][1]+m[j+1][1]+m[j+1][2]+m[j+2][1]+m[j+2][2];
                } else {
                  	fv[features[f]]=m[j][1]+m[j+1][1]+m[j+1][2]+m[j+2][1]+m[j+2][2];
                }
		if(fvc[features[f]]){
		  	fvc[features[f]]++;
		}else{
		  	fvc[features[f]]=1;
		}
              
          }
	}


      }  // end for j log
        //rida+=(fv[features[f]])?(fv[features[f]]/fvc[features[f]])+',':0+','; // kysitav
     } // end for f features


  } // end if
  
 wrida='';

 for(ii=0; ii<fl; ii++){
    if(features[ii].indexOf('_')==-1){
      fv['hold_'+features[ii]]=(fv['hold_'+features[ii]])?(fv['hold_'+features[ii]]/fvc['hold_'+features[ii]]):'';
      wrida+=fv['hold_'+features[ii]]+',';

      fv['seek_'+features[ii]]=(fv['seek_'+features[ii]])?(fv['seek_'+features[ii]]/fvc['seek_'+features[ii]]):'';
      wrida+=fv['seek_'+features[ii]]+',';

    }else{
      wrida+=(fv[features[ii]])?(fv[features[ii]]/fvc[features[ii]])+',':',';
    }
     
 }

 fv.klass=classes[i];
 ds.WriteLine(wrida+classes[i]);

  //WScript.Echo("written instance "+log_id);

} // logs



} // for i // classes

WScript.Echo("Finished processing logs!");

ds.Close();


function massiiv(s){
 sisu=s.split(String.fromCharCode(13,10)); 
 return sisu;
}


function standardiseeri(m){
 s=0;
 mp=m.length;
 for(i=0;i<mp;i++){ 
  s+=m[i];
 }
 k=s/mp;
 sd=0;
 d=0;
 for(i=0;i<mp;i++){ 
  diff=m[i]-k;
  sdiff=diff*diff;
  d+=sdiff;
 }
 stdev=Math.sqrt(d/mp);
 uusm=[];
  for(i=0;i<mp;i++){ 
  uusm[i]=(m[i]-k)/stdev;
 }
uusm[mp]=stdev;
return uusm;
}

function stand(mass){
  ml=mass.length;
  keskmine=(eval(mass.join('+')))/ml;
  erinevuste_ruudud=[];
  for(i=0;i<ml;i++){
    erinevuste_ruudud[i]=(mass[i]-keskmine)*(mass[i]-keskmine);
    mass[i]=mass[i]-keskmine;
  }
  std=Math.sqrt(eval(erinevuste_ruudud.join('+'))/ml);
  for(i=0;i<ml;i++){
    mass[i]=(mass[i]/std);
  }
 return mass[i];
}

function std(data){
 dl=data.length;
 for(sum=0,x=0;x<dl;x++){ 
  sum+=data[x];
 }
 k=sum/dl;
 for(d=0,x=0;x<dl;x++){ 
  diff=data[x]-k;
  d+=diff*diff;
 }
 stdev=Math.sqrt(d/dl);
 return stdev;
}

function getFrequency(arr) {
    var freq = {};
    for (var i=0; i<arr.length;i++) {
        var character = arr[i];
        if (freq[character]) {
           freq[character]++;
        } else {
           freq[character] = 1;
        }
    }

    return freq;
}

function loenda(obk,arr,piir){
 if(parseInt(arr[1])==arr[1]&&arr[1]<hpiir){
    if(obk==arr[0]+""){
	if (fv['hold_'+obk]) {
           fv['hold_'+obk]+=arr[1];
        } else {
           fv['hold_'+obk]=arr[1];
        }
	if(fvc['hold_'+obk]){
	   fvc['hold_'+obk]++;
	}else{
           fvc['hold_'+obk]=1;
	}
      }

 }

}

function csv2arr(cont){
   arr=[];
   rws=cont.split('\r\n');
   for(a=0;a<rws.length;a++){
     arr.push(rws[a].split(','));
   }
  return arr;
}