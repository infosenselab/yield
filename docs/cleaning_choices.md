# Cleaning Choices

## JSC files with issues

Why manual cleaning was necessary.

-   This set had a lot of hidden speakers and individual issues, or conversations that were not dialogues and difficulties matching name tags in the dialogue with those in the headers. Also many individual idiosyncratic issues.

Text modifications

-   "Voice of Camara" and interpreter utterances were removed from the dialogues

removed as they didn't contain at least 6 turns of different roles (by raw `.txt` id)

-   00951
-   00952

removed for being duplicated (by raw `.txt` id)

-   00199
-   00261
-   00414
-   00460
-   00466
-   00528
-   00560
-   00588
-   00590
-   00674
-   00675
-   00676
-   00677
-   00678
-   00686
-   00687
-   00688
-   00689
-   00690
-   00706
-   00718
-   00719
-   00720
-   00721
-   00722
-   00723
-   00742
-   00776
-   00777
-   00810
-   00811
-   00897
-   00899
-   00900
-   00960
-   00964
-   00965
-   00973
-   00981
-   01025
-   01026
-   01056
-   01075
-   01076

removed because it contained many external recordings played within the recording

-   00100
-   00101

removed due the significantly different format

-   raw .txt id: 00508 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/SchmittJW/SchmittJW_7-97.htm

Removed for not containing dialogues:

-   raw .txt id: 00111 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/NASA_HQ/NAF/CoatsML/CoatsML_1-4-08.htm

-   raw .txt id: 00158 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/EdmondsEG/Gallery/index.htm

-   raw .txt id: 00171 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/FoxM/index.htm

-   raw .txt id: 00383 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/LunneyGS/Apollo13.htm

-   raw .txt id: 00420 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/McDougleSC/gallery/index.htm

-   raw .txt id: 00452 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/MoserTL/gallery/index.htm

-   raw .txt id: 00481 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/PippenDL/gallery/index.htm

-   raw .txt id: 00565 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/SlezakTR/gallery/index.htm

-   raw .txt id: 00612 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/WhittleDW/gallery/index.htm

-   raw .txt id: 00937 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/NASA_HQ/Administrators/SplawnJL/SplawnJL_3-27-19.htm

-   raw .txt id: 00972 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/NASA_HQ/NAF/SternA/SternA_4-15-08.htm

-   raw .txt id: 00975 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/NASA_HQ/NAF/ShinJS/ShinJS_6-25-08.htm

-   raw .txt id: 00993 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/NACA/CarlsonHW_bio.htm

-   raw .txt id: 01041 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/NACA/SpritzerEK_bio.htm

-   raw .txt id: 00820 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/Shuttle-Mir/LutomskiMG/LutomskiMG_3-12-98.htm

-   raw .txt id: 00823 https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/Shuttle-Mir/LomanovAV/LomanovAV_5-31-98.htm

-   raw .txt id: 375 split into 2 (since its a compilation of 2)

-   raw .txt id: 1060 (https://historycollection.jsc.nasa.gov/JSCHistoryPortal/history/oral_histories/NASA_HQ/Aviatrix/WASP/WASP_7-18-99.htm) eliminated by for being a duplicate of 1058

    -   also were duplicates of raw .txt id: 1058:
        -   raw .txt id: 1062
        -   raw .txt id: 1065
        -   raw .txt id: 1066
        -   raw .txt id: 1070

## Helpers

### Cleaning

REGEX for replace page numbers in the jfk library docs:

`Automated transcript  Page (\d+) For reference only`

REGEX to comprehensively find and replace a space followed by a dash " -" without the `--- Metadata ---` and `--- Dialogue ---` tags

`(?<!-) -(?!-)`

Spoting page numbers:

-   `^\d+\s*$`
-   `Page\s*?\d+\.?` - when it also has `Page` before it