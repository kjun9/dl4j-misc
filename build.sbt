lazy val sparkVersion = "2.3.2"
lazy val hadoopVersion = "2.8.4"

lazy val core = (project in file ("."))
  .settings(
    name := "dl4j-misc",
    organization := "com.kjun9.example",
    scalaVersion := "2.11.12",
    version := "0.0.1-SNAPSHOT",
    resolvers ++= Seq(
      Resolver.sonatypeRepo("releases"),
      "Spark Packages Repo" at "https://dl.bintray.com/spark-packages/maven/"
    ),
    assemblySettings,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-sql" % sparkVersion % Provided,
      "org.apache.hadoop" % "hadoop-common" % hadoopVersion,
      "org.apache.hadoop" % "hadoop-mapreduce-client-core" % hadoopVersion,
      "org.deeplearning4j" %% "dl4j-spark" % "1.0.0-beta3_spark_2" excludeAll(
        ExclusionRule(organization = "org.apache.spark")
      ),
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta3"
    )
  )

lazy val assemblySettings = Seq(
  assemblyMergeStrategy in assembly := {
    case x if Assembly.isConfigFile(x) =>
      MergeStrategy.concat
    case PathList(ps @ _*)
      if Assembly.isReadme(ps.last) || Assembly.isLicenseFile(ps.last) =>
      MergeStrategy.rename
    case PathList("META-INF", xs @ _*) =>
      xs map { _.toLowerCase } match {
        case "manifest.mf" :: Nil | "index.list" :: Nil |
             "dependencies" :: Nil =>
          MergeStrategy.discard
        case ps @ x :: xs
          if ps.last.endsWith(".sf") || ps.last.endsWith(".dsa") =>
          MergeStrategy.discard
        case "plexus" :: xs =>
          MergeStrategy.discard
        case "services" :: xs =>
          MergeStrategy.filterDistinctLines
        case "spring.schemas" :: Nil | "spring.handlers" :: Nil =>
          MergeStrategy.filterDistinctLines
        case _ => MergeStrategy.first
      }
    case _ => MergeStrategy.first
  },
  logLevel in assembly := Level.Debug
)

