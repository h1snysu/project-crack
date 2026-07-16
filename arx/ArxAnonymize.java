import java.io.File;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.DataHandle;
import org.deidentifier.arx.criteria.KAnonymity;

/**
 * Headless ARX-LR anonymizer: a scriptable replacement for the ARX GUI.
 *
 * Reproduces the paper / cra README setup:
 *   - k-anonymity
 *   - each QI generalized via its hierarchy CSV
 *   - minimum generalization level = 1 (never emit raw layer-0 values)
 *   - local recoding via optimizeIterativeFast (records end at mixed levels)
 *   - suppression allowed (fully-suppressed records -> "*", the CRA outliers)
 *
 * Usage:
 *   java -cp libarx-3.9.2.jar:. ArxAnonymize \
 *       --input in.csv --output out.csv --k 5 \
 *       --qi "Age|Systolic Blood Pressure" \
 *       --hier "hierarchy_age.csv|hierarchy_systolic-blood-pressure.csv" \
 *       --hier-dir ../dataset/min1/hierarchies \
 *       [--suppression 1.0] [--gsfactor 0.5] [--min-level 1] [--no-local]
 */
public class ArxAnonymize {

    static String arg(Map<String,String> m, String k, String def) {
        return m.getOrDefault(k, def);
    }

    public static void main(String[] args) throws Exception {
        Map<String,String> a = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String s = args[i];
            if (s.startsWith("--")) {
                String key = s.substring(2);
                if (key.equals("no-local")) { a.put(key, "true"); }
                else if (i + 1 < args.length) { a.put(key, args[++i]); }
            }
        }

        String input   = a.get("input");
        String output  = a.get("output");
        int k          = Integer.parseInt(arg(a, "k", "5"));
        String[] qis   = a.get("qi").split("\\|");
        String[] hiers = a.get("hier").split("\\|");
        String hierDir = arg(a, "hier-dir", ".");
        double suppression = Double.parseDouble(arg(a, "suppression", "1.0"));
        // gsFactor 0.0 (prefer generalization over suppression) + mode=iter was
        // validated to reproduce the hand-made GUI export to within ~1 EQ.
        double gsFactor    = Double.parseDouble(arg(a, "gsfactor", "0.0"));
        int minLevel       = Integer.parseInt(arg(a, "min-level", "1"));
        // mode: global (no local recoding) | fast | iterfast | iter (finest; default)
        String mode        = arg(a, "mode", a.containsKey("no-local") ? "global" : "iter");

        if (input == null || output == null || a.get("qi") == null || a.get("hier") == null) {
            System.err.println("required: --input --output --k --qi --hier [--hier-dir]");
            System.exit(2);
        }
        if (qis.length != hiers.length) {
            System.err.println("--qi and --hier must have the same number of |-separated entries");
            System.exit(2);
        }

        // Load data (comma-delimited, double-quote quoting is the ARX default).
        Data data = Data.create(new File(input), StandardCharsets.UTF_8, ',');

        // Register QIs with their hierarchies.
        java.util.Set<String> qiSet = new java.util.HashSet<>();
        for (int i = 0; i < qis.length; i++) {
            File hf = new File(hierDir, hiers[i]);
            Hierarchy h = Hierarchy.create(hf, StandardCharsets.UTF_8, ',');
            data.getDefinition().setAttributeType(qis[i], h);
            data.getDefinition().setMinimumGeneralization(qis[i], minLevel);
            data.getDefinition().setMaximumGeneralization(qis[i], h.getHierarchy()[0].length - 1);
            qiSet.add(qis[i]);
        }
        // Keep every non-QI column as-is (insensitive) rather than suppressing
        // it: preserves identifiers/diagnoses like the hand-made GUI export and,
        // crucially, keeps the __fake__ marker so fakes stay traceable post-ARX.
        for (int c = 0; c < data.getHandle().getNumColumns(); c++) {
            String name = data.getHandle().getAttributeName(c);
            if (!qiSet.contains(name)) {
                data.getDefinition().setAttributeType(name, AttributeType.INSENSITIVE_ATTRIBUTE);
            }
        }

        // GUI local-recoding defaults (from WorkerAnonymize.java, ARX 3.9.2):
        //   metric = Loss with gsFactor = 0
        //   suppressionLimit = 1 - 1/numIterations
        //   local recode = optimizeIterativeFast(output, records = 1/numIterations)
        int numIterations = Integer.parseInt(arg(a, "num-iterations", "100"));
        boolean guiLocal = mode.equals("guilocal");
        if (guiLocal) {
            gsFactor = Double.parseDouble(arg(a, "gsfactor", "0.0"));
            suppression = 1.0 - (1.0 / (double) numIterations);
        }

        String metricName = arg(a, "metric", "loss");
        org.deidentifier.arx.metric.Metric<?> metric;
        switch (metricName) {
            case "precision": metric = org.deidentifier.arx.metric.Metric.createPrecisionMetric(); break;
            case "height":    metric = org.deidentifier.arx.metric.Metric.createHeightMetric(); break;
            case "entropy":   metric = org.deidentifier.arx.metric.Metric.createEntropyMetric(); break;
            case "aecs":      metric = org.deidentifier.arx.metric.Metric.createAECSMetric(); break;
            case "disc":      metric = org.deidentifier.arx.metric.Metric.createDiscernabilityMetric(); break;
            case "loss":      default: metric = org.deidentifier.arx.metric.Metric.createLossMetric(gsFactor); break;
        }
        ARXConfiguration config = ARXConfiguration.create(suppression, metric);
        config.addPrivacyModel(new KAnonymity(k));

        ARXAnonymizer anonymizer = new ARXAnonymizer();
        ARXResult result = anonymizer.anonymize(data, config);

        if (!result.isResultAvailable()) {
            System.err.println("ARX: no anonymization solution for k=" + k
                + " within suppression limit " + suppression);
            System.exit(3);
        }

        DataHandle handle = result.getOutput(false);

        // Local recoding: de-generalize records where k still holds, producing
        // mixed generalization levels like ARX-LR. NOTE the ARX API quirk:
        // optimizeFast/optimizeIterativeFast take (handle, records, gsFactor)
        // where `records` is the fraction optimized per step; optimize/
        // optimizeIterative take (handle, gsFactor).
        double records = Double.parseDouble(arg(a, "records", "1.0"));
        org.deidentifier.arx.ARXListener noop = new org.deidentifier.arx.ARXListener() {
            public void progress(double p) {}
        };
        try {
            if (mode.equals("guilocal")) {
                // Exactly replicate the GUI: optimizeIterativeFast(handle,
                // records = 1/numIterations), gsFactor taken from the metric.
                result.optimizeIterativeFast(handle, 1.0 / (double) numIterations, noop);
            } else if (mode.equals("fast")) {
                result.optimizeFast(handle, records, gsFactor, noop);
            } else if (mode.equals("iterfast")) {
                result.optimizeIterativeFast(handle, records, gsFactor, noop);
            } else if (mode.equals("iter")) {
                int maxIter = Integer.parseInt(arg(a, "max-iter", "100"));
                double adaption = Double.parseDouble(arg(a, "adaption", "0.1"));
                result.optimizeIterative(handle, gsFactor, maxIter, adaption);
            } // mode == "global": leave the global-optimum output as-is
        } catch (org.deidentifier.arx.exceptions.RollbackRequiredException e) {
            System.err.println("local recoding rolled back: " + e.getMessage());
        }

        handle.save(new File(output), ',');
        System.err.println("ARX done: rows=" + handle.getNumRows()
            + " k=" + k + " mode=" + mode + " gsFactor=" + gsFactor
            + " suppression=" + suppression);
    }
}
