organization := "pl.edu.icm"

normalizedName := "sparkling-ferns"

version := "0.1.0"

name := "Sparkling Ferns"

organizationName := "Interdisciplinary Centre for Mathematical and Computational Modelling, University of Warsaw"

description := "An implementation of Random Ferns machine learning algorithm for Apache Spark."

organizationHomepage := Some(url("http://www.icm.edu.pl"))

scalaVersion := "2.10.4"

parallelExecution in Test := false

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.0.2"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.0.2"

libraryDependencies += "com.jsuereth" %% "scala-arm" % "1.3" % "test"

libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.1" % "test"

resolvers += "Akka Repository" at "http://repo.akka.io/releases/"

publishMavenStyle := true

publishTo := {
  val nexus = "https://oss.sonatype.org/"
  if (isSnapshot.value)
    Some("snapshots" at nexus + "content/repositories/snapshots")
  else
    Some("releases"  at nexus + "service/local/staging/deploy/maven2")
}

publishArtifact in Test := false

pomIncludeRepository := { _ => false }

licenses := Seq("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.html"))

homepage := Some(url("https://github.com/CeON/sparkling-ferns"))

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