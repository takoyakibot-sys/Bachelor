using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;
using System;

namespace MySolidWorksProject
{
    class Program
    {
        static void Main(string[] args)
        {
            // SolidWorksアプリケーションに接続
            SldWorks swApp = new SldWorks();
            swApp.Visible = true;

            // エラーおよび警告を格納する変数を定義
            int errors = 0;
            int warnings = 0;

            // 新しい部品ドキュメントを開く
            ModelDoc2 swModel = (ModelDoc2)swApp.OpenDoc6(
                @"C:\SWAPI\Part1.SLDPRT", 
                (int)swDocumentTypes_e.swDocPART, 
                (int)swOpenDocOptions_e.swOpenDocOptions_Silent, 
                "", 
                ref errors, 
                ref warnings); // refキーワードを使って変数を渡す

            if (swModel == null)
            {
                Console.WriteLine("Error: 部品ドキュメントが開けませんでした。");
                if (errors != 0)
                {
                    Console.WriteLine($"Errors: {errors}");
                }
                if (warnings != 0)
                {
                    Console.WriteLine($"Warnings: {warnings}");
                }
                return;
            }

            // スケッチを開始
            swModel.SketchManager.InsertSketch(true);

            // 原点に10mm四方の四角形を作成
            swModel.SketchManager.CreateLine(0, 0, 0, 0.01, 0, 0);
            swModel.SketchManager.CreateLine(0.01, 0, 0, 0.01, 0.01, 0);
            swModel.SketchManager.CreateLine(0.01, 0.01, 0, 0, 0.01, 0);
            swModel.SketchManager.CreateLine(0, 0.01, 0, 0, 0, 0);

            // スケッチを終了
            swModel.SketchManager.InsertSketch(false);

            // 押し出しの深さを定義
            double depth = 0.01;  // 10mm

            // 押し出しの作成
            FeatureManager swFeatureManager = swModel.FeatureManager;
            swFeatureManager.FeatureExtrusion2(
                true,    // flip
                false,   // merge
                false,   // useAutoSelect
                (int)swEndConditions_e.swEndCondBlind, // startCondition
                (int)swEndConditions_e.swEndCondBlind, // endCondition
                depth,    // depth (10mm)
                0, // 第二方向の深さはなし
                false,   // draftOutward
                false,   // draftBothDirections
                false,   // thinWall
                false,   // thinWallDirectionOutward
                0,    // wallThickness
                0, // secondWallThickness
                false,   // flipSideToCut
                false,   // reverseSketchOffset
                false,   // autoAdd
                false,   // reverseDirection
                false,   // capEnds
                false,   // useThickCaps
                false,   // removeEndCaps
                0, // capThickness
                0, // capThicknessDepth
                false    // thinWallEndsInward
            );

            // ファイルを保存
            swModel.SaveAs(@"C:\SWAPI\cube10mm.sldprt");

            // 完了メッセージ
            Console.WriteLine("10mmの立方体が作成され、保存されました。");
        }
    }
}
