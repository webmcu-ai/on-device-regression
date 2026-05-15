// ======================================================
// on-device-regression-v003 — CHANGE NOTES AND SURGICAL DIFFS
//
// Three independent changes.  Apply all three for best extrapolation.
// Each change is labelled CHANGE-1 / CHANGE-2 / CHANGE-3.
// Search for the OLD block in your v002 file, replace with the NEW block.
// Nothing else in the file is touched.
//
// Summary of what v002 still got wrong and why extrapolation was capped:
//
//   ISSUE-1: Blank-class distance target was 0.5f.
//     Every blank image pulled myOutput_b[0] toward 0.5 during training.
//     With many blank images the distance head is anchored well below 1.0,
//     so even the top label settles near 0.9–0.95 rather than above 1.0.
//     Fix: change the blank distance target to 0.0f (object is absent,
//     distance is irrelevant — anchor it at zero, not the midpoint).
//
//   ISSUE-2: The output bias is warm-started at 0.5f (line 458).
//     That is a sensible midpoint init for a 0–1 range, but combined
//     with ISSUE-1 it double-anchors the head below 1.0.
//     Fix: init myOutput_b[0] = 0.9f — closer to the top training target.
//     This shortens the distance the optimiser must travel during training
//     and leaves more headroom above 1.0 at convergence.
//
//   ISSUE-3: No calibration — the raw output is multiplied by myDistMax
//     (10.0f) which assumes the network reliably reaches exactly 1.0 for
//     the farthest class.  In practice it reaches ~0.92–0.97, so
//     "10 cm" reads as "9.2–9.7 cm" and anything beyond is suppressed.
//     Fix: after training, run a one-shot calibration pass over the
//     validation (or all training) images, record the mean raw myDistPred
//     for each distance class, then fit a least-squares line through those
//     (rawPred, realDist_cm) anchor points.  At inference, apply
//       myCalibratedDist = myCalibSlope * myDistPred + myCalibOffset
//     instead of myDistPred * myDistMax.
//     The fitted line naturally extrapolates beyond the training maximum.
//
// ======================================================


// ============================================================
// CHANGE-1 — Fix blank-class distance target (ISSUE-1)
// Location: inside myActionTrain(), the training loop, ~line 1155
// ============================================================

// ---- OLD (v002) ----
        bool myIsBlank     = (img.label == 0);
        float myDTarget    = myIsBlank ? 0.5f : myDistTarget[img.label - 1];
        float myConfTarget = myIsBlank ? 0.0f : 1.0f;

// ---- NEW (v003) ----
        bool myIsBlank     = (img.label == 0);
        // CHANGE-1: blank distTarget changed from 0.5f to 0.0f.
        // When confidence=0 the distance head gradient is suppressed by
        // myConfLossWeight but not zeroed.  Targeting 0.0 keeps blank
        // images from anchoring the distance head at the midpoint,
        // giving the top label room to push above 1.0.
        float myDTarget    = myIsBlank ? 0.0f : myDistTarget[img.label - 1];
        float myConfTarget = myIsBlank ? 0.0f : 1.0f;

// Apply the same fix in the VALIDATION PASS (~line 1220):

// ---- OLD (v002) ----
        bool vIsBlank    = (vitem.label == 0);
        float vDTarget   = vIsBlank ? 0.5f : myDistTarget[vitem.label - 1];

// ---- NEW (v003) ----
        bool vIsBlank    = (vitem.label == 0);
        float vDTarget   = vIsBlank ? 0.0f : myDistTarget[vitem.label - 1]; // CHANGE-1


// ============================================================
// CHANGE-2 — Warm-start output bias higher (ISSUE-2)
// Location: myAllocateMemory(), ~line 457
// ============================================================

// ---- OLD (v002) ----
  // Bias init: distance head starts at 0.5 (midpoint), confidence at 0
  myOutput_b[0] = 0.5f;   // distance neuron bias — warm-start near midpoint
  myOutput_b[1] = 0.0f;   // confidence neuron bias

// ---- NEW (v003) ----
  // CHANGE-2: distance bias warm-started at 0.9f instead of 0.5f.
  // The training targets span 0.2–1.0; starting near the top shortens
  // the optimiser's journey and leaves the bias free to exceed 1.0
  // once gradient pressure from the top label pushes it there.
  myOutput_b[0] = 0.9f;   // distance neuron bias — warm-start near top target
  myOutput_b[1] = 0.0f;   // confidence neuron bias


// ============================================================
// CHANGE-3 — Post-training calibration fit + calibrated inference
// (ISSUE-3)
//
// Three parts:
//   3a. New globals (add near other globals, ~line 288)
//   3b. New function myFitCalibration() (add before myActionInfer)
//   3c. Call myFitCalibration() at end of myActionTrain()
//   3d. Use myCalibratedDist in myActionInfer() instead of raw * myDistMax
// ============================================================

// ---- 3a: NEW GLOBALS — add after the existing myDistPred / myConfPred lines ----
// (~line 289, after:  float  myConfPred  = 0.0f; )

// CHANGE-3a: calibration state — filled once after training by myFitCalibration()
float myCalibSlope  = 10.0f;   // default = myDistMax (identity mapping)
float myCalibOffset =  0.0f;
bool  myCalibReady  = false;


// ---- 3b: NEW FUNCTION — insert just before void myActionInfer() ----

// ======================================================
// CALIBRATION FIT (CHANGE-3b)
//
// Call once after training with the mean raw myDistPred values the
// network actually produces for each distance class.
// Fits a least-squares line through (rawPred[], realDist_cm[]) pairs
// and stores slope + offset for use at inference.
//
// With 3 labels the line is over-determined (3 points, 2 params) —
// that is fine; least-squares gives the best straight-line fit.
// The line extrapolates naturally for rawPred > 1.0.
// ======================================================
void myFitCalibration(float myRawPreds[], float myRealDists[], int myN) {
  if (myN < 2) {
    // Fallback: single point or no data — use identity * myDistMax
    myCalibSlope  = myDistMax;
    myCalibOffset = 0.0f;
    myCalibReady  = true;
    Serial.println("Calibration: fallback identity (too few points)");
    return;
  }
  float mySumX = 0, mySumY = 0, mySumXX = 0, mySumXY = 0;
  for (int i = 0; i < myN; i++) {
    mySumX  += myRawPreds[i];
    mySumY  += myRealDists[i];
    mySumXX += myRawPreds[i] * myRawPreds[i];
    mySumXY += myRawPreds[i] * myRealDists[i];
  }
  float myDenom = myN * mySumXX - mySumX * mySumX;
  if (fabsf(myDenom) < 1e-6f) {
    // Degenerate (all predictions identical) — use mean ratio
    myCalibSlope  = (mySumX > 1e-4f) ? mySumY / mySumX : myDistMax;
    myCalibOffset = 0.0f;
  } else {
    myCalibSlope  = (myN * mySumXY - mySumX * mySumY) / myDenom;
    myCalibOffset = (mySumY - myCalibSlope * mySumX)   / myN;
  }
  myCalibReady = true;
  Serial.printf("Calibration fit: dist = %.3f * rawPred + %.3f\n",
                myCalibSlope, myCalibOffset);
  Serial.println("  (extrapolates linearly beyond training max)");
}


// ---- 3c: CALL myFitCalibration() at the end of myActionTrain() ----
// Insert AFTER the existing validation block (after the Serial.printf
// "Validation: MAE=..." line) and BEFORE mySaveWeights().

// ---- OLD (v002) — the end of myActionTrain() looks like: ----
    mySaveWeights();
    myWeightsTrained = true;

// ---- NEW (v003) — replace those two lines with: ----

    // CHANGE-3c: calibration pass — run forward on all training images,
    // accumulate mean raw myDistPred per distance class, then fit line.
    {
      float myCalibSumPred[NUM_DIST_CLASSES] = {};
      int   myCalibCount[NUM_DIST_CLASSES]   = {};

      for (auto& citem : myTrainingData) {
        if (citem.label == 0) continue;                         // skip blank
        if (!myLoadImageFromFile(citem.path.c_str(), myInputBuffer)) continue;
        myForwardPass(myInputBuffer);
        int ci = citem.label - 1;                              // 0-based class index
        myCalibSumPred[ci] += myDistPred;
        myCalibCount[ci]++;
      }

      // Build anchor arrays: mean rawPred → real distance (cm)
      float myAnchorRaw[NUM_DIST_CLASSES];
      float myAnchorDist[NUM_DIST_CLASSES];
      int   myAnchorN = 0;
      for (int ci = 0; ci < NUM_DIST_CLASSES; ci++) {
        if (myCalibCount[ci] > 0) {
          myAnchorRaw[myAnchorN]  = myCalibSumPred[ci] / myCalibCount[ci];
          myAnchorDist[myAnchorN] = myDistTarget[ci] * myDistMax;  // e.g. 0.2*10=2cm
          Serial.printf("  Calib anchor: class %s%s  meanRaw=%.4f  real=%.1f%s\n",
                        myDistLabel[ci].c_str(), myDistUnit.c_str(),
                        myAnchorRaw[myAnchorN], myAnchorDist[myAnchorN],
                        myDistUnit.c_str());
          myAnchorN++;
        }
      }
      myFitCalibration(myAnchorRaw, myAnchorDist, myAnchorN);
    }

    mySaveWeights();
    myWeightsTrained = true;


// ---- 3d: USE myCalibratedDist IN myActionInfer() ----
// There are TWO places in myActionInfer() where myRealDist is computed.
// Replace BOTH occurrences.

// ---- OLD (v002) — first occurrence (~line 1342) ----
      // De-normalize distance to real-world units
      // NOTE: myDistPred can exceed 1.0 — this is intentional (extrapolation)
      float myRealDist = myDistPred * myDistMax;
      bool  myObjectDetected = (myConfPred >= myConfThreshold);

// ---- NEW (v003) ----
      // CHANGE-3d: use calibration line instead of raw * myDistMax.
      // myCalibSlope/Offset were fitted to the actual network outputs after
      // training, so the line passes through the real anchor distances and
      // extrapolates naturally for rawPred > 1.0.
      // Fallback to identity if calibration was never run (e.g. baked weights).
      float myRealDist = myCalibReady
                         ? (myCalibSlope * myDistPred + myCalibOffset)
                         : (myDistPred * myDistMax);
      bool  myObjectDetected = (myConfPred >= myConfThreshold);

// ---- OLD (v002) — second occurrence (~line 1379) ----
    float myRealDist = myDistPred * myDistMax;
    bool  myObjectDetected = (myConfPred >= myConfThreshold);

// ---- NEW (v003) ----
    float myRealDist = myCalibReady                            // CHANGE-3d
                       ? (myCalibSlope * myDistPred + myCalibOffset)
                       : (myDistPred * myDistMax);
    bool  myObjectDetected = (myConfPred >= myConfThreshold);


// ======================================================
// OPTIONAL — Save/load calibration to SD so baked-weight
// builds also get extrapolation.  Add to mySaveWeights()
// and myLoadWeights() if desired.
// ======================================================

// In mySaveWeights(), after the existing f.write() calls:
  // OPTIONAL-3e: persist calibration alongside weights
  f.write((uint8_t*)&myCalibSlope,  sizeof(float));
  f.write((uint8_t*)&myCalibOffset, sizeof(float));

// In myLoadWeights(), after the existing f.read() calls:
  // OPTIONAL-3e: restore calibration
  if (f.read((uint8_t*)&myCalibSlope,  sizeof(float)) == sizeof(float) &&
      f.read((uint8_t*)&myCalibOffset, sizeof(float)) == sizeof(float)) {
    myCalibReady = true;
    Serial.printf("Calibration loaded: slope=%.3f offset=%.3f\n",
                  myCalibSlope, myCalibOffset);
  }

// NOTE: if you add OPTIONAL-3e you must retrain once to write the new
// binary format.  Old myWeights.bin files will load weights correctly
// but the trailing calib read will fail silently (f.read returns 0),
// myCalibReady stays false, and inference falls back to raw * myDistMax.
// ======================================================
