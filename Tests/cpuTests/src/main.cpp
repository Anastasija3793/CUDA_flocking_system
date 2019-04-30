#include <gtest/gtest.h>
#include <ngl/Vec3.h>

#include "Boid.h"
#include "Flock.h"


int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(Boid, pos_and_vel)
{
    auto startPos = ngl::Vec3(0.0f,0.0f,0.0f);
    auto startVel = ngl::Vec3(1.0f,0.0f,0.0f);
    Boid b(startPos,startVel);

    EXPECT_EQ(b.m_pos.m_x , 0.0f);
    EXPECT_EQ(b.m_pos.m_y , 0.0f);
    EXPECT_EQ(b.m_pos.m_z , 0.0f);

    EXPECT_EQ(b.m_vel.m_x , 1.0f);
    EXPECT_EQ(b.m_vel.m_y , 0.0f);
    EXPECT_EQ(b.m_vel.m_z , 0.0f);
}

TEST(Boid, update)
{
    ngl::Vec3 startPos = ngl::Vec3(1.0f,1.0f,1.0f);
    ngl::Vec3 startVel = ngl::Vec3(1.0f,1.0f,1.0f);

    Boid b(startPos,startVel);

    ngl::Vec3 testPos = ngl::Vec3(1.0f,1.0f,1.0f);
    ngl::Vec3 testVel = ngl::Vec3(1.0f,1.0f,1.0f);
    ngl::Vec3 testAcc = ngl::Vec3(0.0f,0.0f,0.0f);
    float maxSpeed = 1.0f;

    auto newVel = testVel + testAcc;
    if(newVel.length() > maxSpeed)
    {
        newVel = (newVel/newVel.length())*maxSpeed;
    }
    ngl::Vec3 newPos = testPos + newVel;
    testAcc*=maxSpeed;

    b.update();


    EXPECT_EQ(b.m_vel.m_x, newVel.m_x);
    EXPECT_EQ(b.m_vel.m_y, newVel.m_y);
    EXPECT_EQ(b.m_vel.m_z, newVel.m_z);

    EXPECT_EQ(b.m_pos.m_x, newPos.m_x);
    EXPECT_EQ(b.m_pos.m_y, newPos.m_y);
    EXPECT_EQ(b.m_pos.m_z, newPos.m_z);

    EXPECT_TRUE(b.m_pos == newPos);

}

TEST(Boid, seek)
{
    ngl::Vec3 startPos = ngl::Vec3(1.0f,1.0f,1.0f);
    ngl::Vec3 startVel = ngl::Vec3(1.0f,1.0f,1.0f);
    ngl::Vec3 target = ngl::Vec3(0.0f,0.0f,0.0f);
    float maxSpeed = 1.0f;
    float maxForce = 0.03f;

    Boid b(startPos,startVel);

    ngl::Vec3 desired = target - startPos;
    desired.normalize();
    desired*=maxSpeed;

    target = desired - startVel;

    if(target.length() > maxForce)
    {
        target = (target/target.length())*maxForce;
    }

    b.seek(target);


    EXPECT_TRUE(b.m_pos == startPos);
    EXPECT_TRUE(b.m_vel == startVel);
    EXPECT_TRUE(b.m_desired == desired);
    EXPECT_TRUE(b.m_target == target);
}

TEST(Boid, separate)
{
    ngl::Vec3 startPos = ngl::Vec3(1.0f,1.0f,1.0f);
    ngl::Vec3 startVel = ngl::Vec3(1.0f,1.0f,1.0f);
    float maxSpeed = 1.0f;
    float maxForce = 0.03f;

    ngl::Vec3 sep = ngl::Vec3(0.0f,0.0f,0.0f);
    float sepRad = 15.0f;
    Boid b(startPos,startVel);

    int neighbours = 0;
    std::vector<Boid*> near;

    for(unsigned int i=0; i<near.size(); ++i)
    {
        float dist = (startPos - near[i]->m_pos).length();
        if((dist>0) && (dist<sepRad))
        {
            ngl::Vec3 diff = startPos - near[i]->m_pos;
            diff.normalize();
            diff/=dist;
            sep+=diff;
            neighbours++;
        }
    }

    if(neighbours>0)
    {
        sep/=(float(neighbours));
        //m_steer/=count;
    }
    if(sep.length()>0)
    {
        sep.normalize();
        sep*=maxSpeed;
        sep-=startVel;
        //limit by max_force
        if(sep.length() > maxForce)
        {
            sep = (sep/sep.length())*maxForce;
        }
    }

    b.separate(sep);

    EXPECT_TRUE(b.m_pos == startPos);
    EXPECT_TRUE(b.m_vel == startVel);
    EXPECT_TRUE(b.m_sep == sep);
}


//TEST(test, cpu)
//{
//    Flock f_cpu(10);

//    for(int i=0; i<f_cpu.m_numBoids; i++)
//    {
//        f_cpu.update();

//        EXPECT_TRUE(f_cpu.)
//    }
//}
