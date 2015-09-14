lazy val commonSettings = Seq(
    organization := "pl.edu.icm",
    normalizedName := "sparkling-ferns",
    version := "0.2.0-SNAPSHOT",
    name := "Sparkling Ferns",
    organizationName := "Interdisciplinary Centre for Mathematical and Computational Modelling, University of Warsaw",
    description := "An implementation of Random Ferns machine learning algorithm for Apache Spark.",
    organizationHomepage := Some(url("http://www.icm.edu.pl")),
    scalaVersion := "2.10.4"
)

lazy val root = (project in file(".")).
  settings(commonSettings).
  settings(
    parallelExecution in Test := false,
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "1.0.2" % "provided",
      "org.apache.spark" %% "spark-mllib" % "1.0.2" % "provided",
      "com.jsuereth" %% "scala-arm" % "1.3" % "test",
      "org.scalatest" %% "scalatest" % "2.2.1" % "test"
    ),
    resolvers += "Akka Repository" at "http://repo.akka.io/releases/",
    publishMavenStyle := true,
    publishTo := {
      val nexus = "https://oss.sonatype.org/"
      if (isSnapshot.value)
        Some("snapshots" at nexus + "content/repositories/snapshots")
      else
        Some("releases"  at nexus + "service/local/staging/deploy/maven2")
    },
    publishArtifact in Test := false,
    pomIncludeRepository := { _ => false },
    test in assembly := {},
    licenses := Seq("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.html")),
    homepage := Some(url("https://github.com/CeON/sparkling-ferns")),
    pomExtra := (
        <scm>
          <url>git@github.com:CeON/sparkling-ferns.git</url>
          <connection>scm:git:git@github.com:CeON/sparkling-ferns.git</connection>
        </scm>
        <developers>
          <developer>
            <id>matfed</id>
            <name>Mateusz Fedoryszak</name>
            <email>mfedoryszak@gmail.com</email>
          </developer>
        </developers>)
  )


